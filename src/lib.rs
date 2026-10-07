//! Silero TTS v5 as a DLL. LoreLens itself stays free of libtorch (300 MB of libraries
//! built against another C runtime): it loads this DLL from the voice's folder when the
//! voice is installed, and talks to it through the few C functions below.
//!
//! Strings are UTF-8 and end with a zero byte. A function that fails leaves its reason
//! for `silero_error`.

mod silero;

use std::cell::RefCell;
use std::ffi::{c_char, c_void, CStr};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;

pub use silero::{Silero, SAMPLE_RATE};

thread_local! {
    static LAST_ERROR: RefCell<String> = const { RefCell::new(String::new()) };
}

/// How many processor threads the networks may use.
pub fn set_threads(threads: i32) {
    tch::set_num_threads(threads.max(1));
}

/// Gets samples at 24 kHz as they are ready; returns `false` to stop the reading.
pub type OnSamples = extern "C" fn(context: *mut c_void, samples: *const f32, count: usize) -> bool;

/// Runs `work`, turning an error or a panic into `None` and a message for `silero_error`.
fn guarded<T>(work: impl FnOnce() -> anyhow::Result<T>) -> Option<T> {
    let outcome = catch_unwind(AssertUnwindSafe(work)).unwrap_or_else(|_| Err(anyhow::anyhow!("внутренняя ошибка голоса Silero")));
    match outcome {
        Ok(value) => Some(value),
        Err(e) => {
            LAST_ERROR.with(|last| *last.borrow_mut() = format!("{e:#}"));
            None
        }
    }
}

/// # Safety
/// `text` is null or points to a zero-terminated string.
unsafe fn string(text: *const c_char) -> anyhow::Result<String> {
    anyhow::ensure!(!text.is_null(), "пустой указатель вместо строки");
    Ok(CStr::from_ptr(text).to_str()?.to_string())
}

/// Copies `text` into `buffer` (cut to fit, always zero-terminated) and returns its full length.
///
/// # Safety
/// `buffer` is null or has room for `size` bytes.
unsafe fn copy_out(text: &str, buffer: *mut c_char, size: usize) -> usize {
    if !buffer.is_null() && size > 0 {
        let count = text.len().min(size - 1);
        std::ptr::copy_nonoverlapping(text.as_ptr(), buffer as *mut u8, count);
        *buffer.add(count) = 0;
    }
    text.len()
}

/// Loads the model from `dir` and warms it up; `threads` is how many processor threads
/// it may use. Returns null on failure.
///
/// # Safety
/// `dir` points to a zero-terminated string.
#[no_mangle]
pub unsafe extern "C" fn silero_open(dir: *const c_char, threads: i32) -> *mut Silero {
    guarded(|| {
        let dir = string(dir)?;
        set_threads(threads);
        let silero = Silero::load(Path::new(&dir))?;
        silero.warm_up();
        Ok(Box::into_raw(Box::new(silero)))
    })
    .unwrap_or(std::ptr::null_mut())
}

/// # Safety
/// `silero` came from `silero_open` and is not used afterwards.
#[no_mangle]
pub unsafe extern "C" fn silero_close(silero: *mut Silero) {
    if !silero.is_null() {
        drop(Box::from_raw(silero));
    }
}

/// The voices, separated by commas. Returns the length of the whole list.
///
/// # Safety
/// `silero` came from `silero_open`; `buffer` has room for `size` bytes.
#[no_mangle]
pub unsafe extern "C" fn silero_voices(silero: *const Silero, buffer: *mut c_char, size: usize) -> usize {
    let Some(silero) = silero.as_ref() else { return 0 };
    copy_out(&silero.voices().join(","), buffer, size)
}

/// Speaks `text`, handing the sound to `on_samples` piece by piece. Returns `false` on failure.
///
/// # Safety
/// `silero` came from `silero_open`; `text` and `voice` point to zero-terminated strings;
/// `on_samples` may be called with `context` until this function returns.
#[no_mangle]
pub unsafe extern "C" fn silero_speak(
    silero: *const Silero,
    text: *const c_char,
    voice: *const c_char,
    speed: f32,
    on_samples: OnSamples,
    context: *mut c_void,
) -> bool {
    guarded(|| {
        let silero = silero.as_ref().ok_or_else(|| anyhow::anyhow!("модель не загружена"))?;
        let (text, voice) = (string(text)?, string(voice)?);
        silero.speak(&text, &voice, speed, &mut |samples, _| on_samples(context, samples.as_ptr(), samples.len()))
    })
    .is_some()
}

/// Why the last call on this thread failed. Returns the length of the whole message.
///
/// # Safety
/// `buffer` has room for `size` bytes.
#[no_mangle]
pub unsafe extern "C" fn silero_error(buffer: *mut c_char, size: usize) -> usize {
    LAST_ERROR.with(|last| copy_out(&last.borrow(), buffer, size))
}
