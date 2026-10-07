//! Silero TTS v5 (Russian) without Python: the three TorchScript networks of the model run
//! through libtorch, and the Python code that sat around them is redone here — cleaning
//! the text, placing stresses, telling homographs apart, marking questions.
//!
//! The model's files come from `export_silero.py`. Each step follows the Python original closely,
//! so the same text gets the same stresses and the same sound.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};
use regex::Regex;
use serde::Deserialize;
use tch::{CModule, IValue, Kind, Tensor};
use unicode_general_category::{get_general_category, GeneralCategory as Cat};

pub const SAMPLE_RATE: u32 = 24_000;
const VOWELS: &str = "аоуыэиеяёю";
const STRESS: char = '+';
/// Characters the accentor treats as the space between words.
const WORD_BREAKS: &str = ".,!?;:<>=()/\\";
/// The model reads a piece of text in one go; longer pieces are cut at sentences.
const PIECE_CHARS: usize = 250;

#[derive(Deserialize)]
struct Tokens {
    unk: i64,
    cls: i64,
    sep: i64,
    pad: i64,
    homo_start: i64,
    homo_end: i64,
}

#[derive(Deserialize)]
struct Data {
    symbols: HashMap<char, i64>,
    speakers: HashMap<String, i64>,
    /// word -> (where the stress mark goes, which letter becomes "ё" or -1)
    exceptions: HashMap<String, (i64, i64)>,
    /// word -> its readings with "+" before the stressed vowel
    homographs: HashMap<String, Vec<String>>,
    vocab: HashMap<String, i64>,
    never_split: HashSet<String>,
    tokens: Tokens,
}

pub struct Silero {
    tts: CModule,
    accentor: CModule,
    homosolver: CModule,
    data: Data,
    tag_question: Regex,
    or_question: Regex,
}

fn is_cyrillic(c: char) -> bool {
    matches!(c, 'а'..='я' | 'А'..='Я' | 'ё' | 'Ё')
}

fn lower(text: &str) -> String {
    text.chars().flat_map(char::to_lowercase).collect()
}

fn collapse_spaces(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

impl Silero {
    pub fn load(dir: &Path) -> Result<Self> {
        let module = |name: &str| {
            let mut module = CModule::load(dir.join(name)).with_context(|| format!("не удалось загрузить {name}"))?;
            module.set_eval();
            Ok::<_, anyhow::Error>(module)
        };
        let data = std::fs::read_to_string(dir.join("data.json")).context("нет data.json")?;
        let mut data: Data = serde_json::from_str(&data).context("data.json не читается")?;
        for readings in data.homographs.values_mut() {
            readings.sort();
        }
        let tags = [
            r"[,\s]+(правда|верно|да)\s*\?$",
            r"[,\s]+(не\s+так\s+ли|не\s+правда\s+ли|разве\s+не\s+так|ведь\s+так)\s*\?$",
            r"[,\s]+ведь\s*\?$",
            r"[,\s]+а\s*\?$",
        ];
        Ok(Self {
            tts: module("tts.jit")?,
            accentor: module("accentor.jit")?,
            homosolver: module("homosolver.jit")?,
            data,
            tag_question: Regex::new(&format!("(?i){}", tags.map(|t| format!("(?:{t})")).join("|")))?,
            or_question: Regex::new(r"(?i)\bили\b")?,
        })
    }

    pub fn voices(&self) -> Vec<String> {
        let mut voices: Vec<_> = self.data.speakers.iter().collect();
        voices.sort_by_key(|(_, id)| **id);
        voices.into_iter().map(|(name, _)| name.clone()).collect()
    }

    /// Speaks `text` piece by piece. `on_piece` gets mono samples at 24 kHz and the piece
    /// as the model read it, with a "+" before every stressed vowel; it returns `false`
    /// to stop. An unknown voice reads as the first one.
    pub fn speak(&self, text: &str, voice: &str, speed: f32, on_piece: &mut dyn FnMut(Vec<f32>, &str) -> bool) -> Result<()> {
        let speaker = self.data.speakers.get(voice).copied().unwrap_or(0);
        let mut got_any = false;
        for piece in pieces(&spell_out(text)) {
            let Some((audio, accented)) = self.speak_piece(&piece, speaker, speed)? else { continue };
            got_any = true;
            if !on_piece(audio, &accented) {
                return Ok(());
            }
        }
        if !got_any {
            bail!("в тексте нет русских слов");
        }
        Ok(())
    }

    /// The first runs of a freshly loaded model are slow: libtorch tunes the networks
    /// while they work. Spending them here keeps the first real reading quick.
    pub fn warm_up(&self) {
        for phrase in ["Проверка связи.", "Раз, два, три — слышно ли меня?"] {
            let _ = self.speak(phrase, "", 1.0, &mut |_, _| true);
        }
    }

    /// `None` when the piece has nothing to read in it.
    fn speak_piece(&self, text: &str, speaker: i64, speed: f32) -> Result<Option<(Vec<f32>, String)>> {
        let clean = self.clean(text);
        if !clean.chars().any(|c| matches!(c, 'а'..='я')) {
            return Ok(None);
        }
        let accented = self.accent(&clean)?;

        // "|" opens the text and "~" closes it.
        let sequence: Vec<i64> = std::iter::once('|')
            .chain(accented.chars())
            .chain(std::iter::once('~'))
            .map(|c| self.data.symbols.get(&c).copied().ok_or_else(|| anyhow!("модель не знает символ «{c}»")))
            .collect::<Result<_>>()?;
        let len = sequence.len() as i64;
        let rates = Tensor::from_slice(&vec![speed; sequence.len()]).view([1, len]);
        let pitches = Tensor::from_slice(&vec![1f32; sequence.len()]).view([1, len]);
        let types = Tensor::from_slice(&self.sentence_types(text, sequence.len())).view([1, len]);

        // The arguments of the model's `forward`, in its order.
        let output = tch::no_grad(|| {
            self.tts.forward_is(&[
                IValue::Tensor(Tensor::from_slice(&sequence).view([1, len])),
                IValue::Tensor(Tensor::from_slice(&[speaker])),
                IValue::Int(SAMPLE_RATE as i64),
                IValue::None, // symb_durs: pauses set by hand
                IValue::Tensor(rates),
                IValue::Tensor(pitches),
                IValue::None, // gt_durs
                IValue::None, // gt_pitch
                IValue::String("cpu".into()),
                IValue::Int(-1), // accent_word
                IValue::Bool(false), // question: superseded by type_ids
                IValue::Tensor(types),
                IValue::None, // focus_mask
            ])
        })
        .context("модель не смогла прочитать текст")?;
        let audio = match output {
            IValue::Tuple(mut parts) if !parts.is_empty() => match parts.swap_remove(0) {
                IValue::Tensor(audio) => audio,
                other => bail!("модель вернула не звук: {other:?}"),
            },
            other => bail!("модель вернула не звук: {other:?}"),
        };
        let samples = Vec::<f32>::try_from(audio.to_kind(Kind::Float).flatten(0, -1))?;
        Ok(Some((samples, accented)))
    }

    /// The text as the model takes it: lower case, only its own letters and punctuation.
    fn clean(&self, text: &str) -> String {
        // A line break ends a sentence unless one has just ended.
        let chars: Vec<char> = text.trim().chars().collect();
        let mut joined = String::new();
        for (at, &c) in chars.iter().enumerate() {
            match c {
                '\n' if at > 0 && ",;:.!?".contains(chars[at - 1]) => joined.push(' '),
                '\n' => joined.push_str(". "),
                c => joined.push(c),
            }
        }
        let kept: String = lower(&joined)
            .chars()
            .map(|c| match c {
                '—' => '–',
                '‑' => '-',
                c => c,
            })
            .filter(|c| !matches!(c, '_' | '~' | '|') && self.data.symbols.contains_key(c))
            .collect();
        collapse_spaces(&kept)
    }

    // ----------------------------------------------------------------- stresses

    /// Puts "+" before stressed vowels and restores "ё": homographs first, by their
    /// context, then every other word.
    pub fn accent(&self, sentence: &str) -> Result<String> {
        self.accent_words(&self.solve_homographs(sentence)?)
    }

    /// Words spelt alike and stressed differently ("за́мок" and "замо́к"): a small BERT
    /// picks the reading from the sentence around the word.
    fn solve_homographs(&self, sentence: &str) -> Result<String> {
        let chars: Vec<char> = sentence.chars().collect();
        let in_word = |c: char| is_cyrillic(c) || c == STRESS;
        let tokens = &self.data.tokens;

        // (start, end, word) of every homograph, and the sentence with the word marked.
        let mut found = Vec::new();
        let mut batch: Vec<Vec<i64>> = Vec::new();
        let mut at = 0;
        while at < chars.len() {
            if !in_word(chars[at]) {
                at += 1;
                continue;
            }
            let start = at;
            while at < chars.len() && in_word(chars[at]) {
                at += 1;
            }
            let word: String = chars[start..at].iter().collect();
            if !self.data.homographs.contains_key(&lower(&word)) {
                continue;
            }
            let before: String = chars[..start].iter().collect();
            let after: String = chars[at..].iter().collect();
            batch.push(self.encode(&format!("{before} [HOMO] {word} [/HOMO] {after}")));
            found.push((start, at, word));
        }
        if found.is_empty() {
            return Ok(sentence.to_string());
        }

        let place = |ids: &[i64], token: i64| ids.iter().position(|&id| id == token).unwrap_or(0) as i64;
        let starts: Vec<i64> = batch.iter().map(|ids| place(ids, tokens.homo_start)).collect();
        let ends: Vec<i64> = batch.iter().map(|ids| place(ids, tokens.homo_end)).collect();
        let width = batch.iter().map(Vec::len).max().unwrap_or(0);
        let padded: Vec<i64> = batch.iter().flat_map(|ids| ids.iter().copied().chain(std::iter::repeat(tokens.pad)).take(width)).collect();
        let input = Tensor::from_slice(&padded).view([batch.len() as i64, width as i64]);
        let logits = tch::no_grad(|| self.homosolver.forward_ts(&[input, Tensor::from_slice(&starts), Tensor::from_slice(&ends)]))
            .context("модель омографов не сработала")?;
        let picks = Vec::<f32>::try_from(logits.sigmoid().round().to_kind(Kind::Float).flatten(0, -1))?;

        let mut out = chars;
        let mut offset = 0;
        for ((start, end, word), pick) in found.into_iter().zip(picks) {
            let readings = &self.data.homographs[&lower(&word)];
            let reading: Vec<char> = readings[(pick as usize).min(readings.len() - 1)].chars().collect();
            let stress_at = reading.iter().position(|&c| c == STRESS).unwrap_or(0);
            // The reading in the letter case of the word in the text.
            let mut cased: Vec<char> = word
                .chars()
                .zip(reading.iter().filter(|&&c| c != STRESS))
                .flat_map(|(from, &to)| if from.is_lowercase() { to.to_lowercase().collect::<Vec<_>>() } else { to.to_uppercase().collect() })
                .collect();
            cased.insert(stress_at.min(cased.len()), STRESS);
            out.splice(start + offset..end + offset, cased);
            offset += 1;
        }
        Ok(out.into_iter().collect())
    }

    /// Every other word: a dictionary of exceptions, then a network that looks at the
    /// letters of the word alone.
    fn accent_words(&self, sentence: &str) -> Result<String> {
        // Words, and the punctuation between them as tokens of its own; a hyphenated
        // word is several tokens, each keeping its hyphen.
        let (mut raw, mut clean, mut wanted) = (Vec::new(), Vec::new(), Vec::new());
        for piece in split_keeping(sentence, |c| c.is_whitespace() || WORD_BREAKS.contains(c)) {
            let parts: Vec<&str> = piece.split('-').collect();
            for (at, part) in parts.iter().enumerate() {
                let last = at + 1 == parts.len();
                let token = if last { part.to_string() } else { format!("{part}-") };
                let letters: String = lower(&token).chars().filter(|c| is_cyrillic(*c)).collect();
                // "что-то", "кому-то": the particle has no stress.
                let particle = last && parts.len() > 1 && *part == "то";
                wanted.push(!letters.is_empty() && !particle);
                clean.push(letters);
                raw.push(token);
            }
        }

        let output = tch::no_grad(|| self.accentor.forward_is(&[IValue::StringList(clean.clone())])).context("модель ударений не сработала")?;
        let IValue::Tuple(parts) = output else { bail!("модель ударений вернула не то") };
        let [IValue::Tensor(stress), IValue::Tensor(yo)] = &parts[..] else { bail!("модель ударений вернула не то") };
        let best = |logits: &Tensor| -> Result<(Vec<i64>, Vec<f32>)> {
            let probs = logits.softmax(1, Kind::Float);
            let (top, index) = probs.max_dim(1, false);
            Ok((Vec::<i64>::try_from(index)?, Vec::<f32>::try_from(top)?))
        };
        let (stress_at, stress_sure) = best(stress)?;
        let (yo_at, yo_sure) = best(yo)?;

        let mut out = String::new();
        for (at, word) in raw.iter().enumerate() {
            if !wanted[at] {
                out.push_str(word);
                continue;
            }
            let mut chars: Vec<char> = word.chars().collect();
            let lowered: Vec<char> = lower(word).chars().collect();
            let has_stress = lowered.contains(&STRESS);
            let has_yo = lowered.contains(&'ё');
            if has_stress && has_yo {
                out.push_str(word);
                continue;
            }
            if has_yo {
                // "ё" is always stressed.
                let places: Vec<usize> = lowered.iter().enumerate().filter(|(_, c)| **c == 'ё').map(|(i, _)| i).collect();
                for (shift, place) in places.into_iter().enumerate() {
                    chars.insert(place + shift, STRESS);
                }
                out.extend(chars);
                continue;
            }
            let yo_cased = |c: char| if c.is_lowercase() { 'ё' } else { 'Ё' };
            if let Some(&(stress_place, yo_place)) = self.data.exceptions.get(&clean[at]) {
                if has_stress {
                    let marks: Vec<usize> = chars.iter().enumerate().filter(|(_, c)| **c == STRESS).map(|(i, _)| i).collect();
                    chars.retain(|c| *c != STRESS);
                    if yo_place >= 0 && marks.contains(&(yo_place as usize + 1)) && (yo_place as usize) < chars.len() {
                        chars[yo_place as usize] = yo_cased(chars[yo_place as usize]);
                    }
                    for mark in marks {
                        chars.insert(mark.min(chars.len()), STRESS);
                    }
                } else {
                    if yo_place >= 0 && (yo_place as usize) < chars.len() {
                        chars[yo_place as usize] = yo_cased(chars[yo_place as usize]);
                    }
                    chars.insert((stress_place.max(0) as usize).min(chars.len()), STRESS);
                }
                out.extend(chars);
                continue;
            }

            // The network answers with the number of the stressed vowel, and with the
            // number of the "е" that is really "ё" (counting from 1; 0 is "none").
            let mut set_stress = stress_sure[at] > 0.5 && !has_stress;
            let set_yo = yo_sure[at] > 0.5;
            let stressed_vowels: Vec<usize> = if has_stress {
                let word: String = lowered.iter().collect();
                word.split(STRESS).map(|part| part.chars().filter(|c| VOWELS.contains(*c)).count()).collect()
            } else {
                vec![stress_at[at] as usize]
            };
            let vowel_places: Vec<usize> = lowered.iter().enumerate().filter(|(_, c)| VOWELS.contains(**c)).map(|(i, _)| i).collect();
            let ye_places: Vec<usize> = lowered.iter().enumerate().filter(|(_, c)| **c == 'е').map(|(i, _)| i).collect();
            let mut stress_places: Vec<usize> = stressed_vowels.iter().filter_map(|&n| vowel_places.get(n).copied()).collect();
            let yo_place = (yo_at[at] > 0).then(|| ye_places.get(yo_at[at] as usize - 1).copied()).flatten();
            if vowel_places.is_empty() {
                out.push_str(word);
                continue;
            }
            if let Some(place) = yo_place.filter(|place| set_yo && stress_places.contains(place) && lowered[*place] == 'е') {
                chars[place] = yo_cased(chars[place]);
            }
            if vowel_places.len() == 1 {
                stress_places = vec![vowel_places[0]];
                set_stress = true;
            }
            if !has_stress && set_stress {
                for (shift, place) in stress_places.into_iter().enumerate() {
                    chars.insert(place + shift, STRESS);
                }
            }
            out.extend(chars);
        }
        Ok(out)
    }

    // ---------------------------------------------------------------- tokenizer

    /// BERT's own way of cutting text into the pieces of its vocabulary.
    fn encode(&self, text: &str) -> Vec<i64> {
        let tokens = &self.data.tokens;
        let spaced: String = text
            .chars()
            .filter(|&c| c != '\0' && c != '\u{fffd}' && !is_control(c))
            .map(|c| if is_space(c) { ' ' } else { c })
            .collect();
        let mut ids = vec![tokens.cls];
        for token in spaced.split_whitespace() {
            let words = if self.data.never_split.contains(token) { vec![token.to_string()] } else { split_keeping(token, is_punctuation) };
            for word in words {
                self.wordpieces(&word, &mut ids);
            }
        }
        ids.push(tokens.sep);
        ids
    }

    /// The longest pieces of the vocabulary the word is made of, from left to right.
    fn wordpieces(&self, word: &str, ids: &mut Vec<i64>) {
        let chars: Vec<char> = word.chars().collect();
        if chars.len() > 100 {
            ids.push(self.data.tokens.unk);
            return;
        }
        let mut pieces = Vec::new();
        let mut start = 0;
        while start < chars.len() {
            let found = (start + 1..=chars.len()).rev().find_map(|end| {
                let piece: String = chars[start..end].iter().collect();
                let piece = if start > 0 { format!("##{piece}") } else { piece };
                self.data.vocab.get(&piece).map(|&id| (id, end))
            });
            let Some((id, end)) = found else {
                ids.push(self.data.tokens.unk);
                return;
            };
            pieces.push(id);
            start = end;
        }
        ids.extend(pieces);
    }

    // ---------------------------------------------------------------- questions

    /// For every symbol of the model's input, the kind of sentence it stands in: the
    /// model raises its voice differently for different questions.
    fn sentence_types(&self, text: &str, len: usize) -> Vec<i64> {
        let sentences = split_sentences(text.trim());
        let kinds: Vec<i64> = sentences.iter().filter(|s| !s.trim().is_empty()).map(|s| self.sentence_kind(s)).collect();
        let first = kinds.first().copied().unwrap_or(0);
        let mut per_char = Vec::new();
        for (at, sentence) in sentences.iter().enumerate() {
            let kind = kinds.get(at).or(kinds.last()).copied().unwrap_or(0);
            per_char.extend(std::iter::repeat(kind).take(sentence.chars().count() + usize::from(at + 1 < sentences.len())));
        }
        if per_char.is_empty() {
            return vec![0; len];
        }
        // The opening "|" takes the kind of the first sentence, and so does any tail.
        std::iter::once(first).chain(per_char).chain(std::iter::repeat(first)).take(len).collect()
    }

    /// 0 statement, 1 "кто/что/где…" question, 2 yes-or-no question, 3 "или" question,
    /// 4 tag question ("…, правда?"), 5 exclamation.
    fn sentence_kind(&self, sentence: &str) -> i64 {
        let clean = strip_quotes(sentence);
        if clean.is_empty() {
            return 0;
        }
        let mut tail = clean;
        if tail.ends_with("?!") || tail.ends_with("?..") || tail.ends_with("?...") {
            tail = tail.trim_end_matches(['.', '!']);
        }
        if tail.ends_with('?') {
            let question = collapse_spaces(&strip_quotes(tail).replace('+', "").replace('ё', "е").replace('Ё', "Е"));
            if self.tag_question.is_match(&question) {
                return 4;
            }
            if has_question_word(&question) {
                return 1;
            }
            return if self.or_question.is_match(&question) { 3 } else { 2 };
        }
        if clean.ends_with('!') {
            5
        } else {
            0
        }
    }
}

/// `text` cut where `is_break` holds, the breaks kept as pieces of their own, in order.
/// Like Python's `re.split` with a group, an empty piece stands before a leading break.
fn split_keeping(text: &str, is_break: impl Fn(char) -> bool) -> Vec<String> {
    let mut out = vec![String::new()];
    let mut in_break = false;
    for c in text.chars() {
        if is_break(c) != in_break {
            in_break = !in_break;
            out.push(String::new());
        }
        out.last_mut().unwrap().push(c);
    }
    out.retain(|piece| !piece.is_empty());
    out
}

/// Sentences: cut at the space after ".", "!" or "?".
fn split_sentences(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = vec![String::new()];
    let mut at = 0;
    while at < chars.len() {
        if chars[at].is_whitespace() && at > 0 && ".!?".contains(chars[at - 1]) {
            while at < chars.len() && chars[at].is_whitespace() {
                at += 1;
            }
            out.push(String::new());
            continue;
        }
        out.last_mut().unwrap().push(chars[at]);
        at += 1;
    }
    out
}

fn strip_quotes(text: &str) -> &str {
    text.trim().trim_start_matches(|c: char| "\"«“„".contains(c) || c.is_whitespace()).trim_end_matches(|c: char| "\"»”’".contains(c) || c.is_whitespace())
}

/// Does the question open with "кто", "где", "какой"…, small talk before it aside?
fn has_question_word(text: &str) -> bool {
    const FILLERS: &[&str] =
        &["а", "ну", "и", "так", "вот", "слышь", "слушай", "скажите", "скажи", "пожалуйста", "вобще", "вообще", "типа", "короче"];
    const QUESTION_WORDS: &str = "кто кого кому кем ком что чего чему чем чём чё чо где куда откуда когда почему зачем как почём отчего \
        насколько сколько скольких скольким сколькими какой какая какое какие какого каких какому каким какими каком какую чей чья чьё \
        чье чьи чьего чьей чьих чьему чьим чьими чьём чьем чью который которая которое которые которого которой которых которому \
        которым которыми котором которую каков какова каково каковы";
    let lowered = lower(text);
    lowered
        .split(|c: char| !(is_cyrillic(c) || c.is_ascii_lowercase()))
        .filter(|word| !word.is_empty())
        .take(8)
        .filter(|word| !FILLERS.contains(word))
        .take(4)
        .any(|word| QUESTION_WORDS.split_whitespace().any(|known| known == word))
}

fn is_control(c: char) -> bool {
    !matches!(c, '\t' | '\n' | '\r')
        && matches!(get_general_category(c), Cat::Control | Cat::Format | Cat::Surrogate | Cat::PrivateUse | Cat::Unassigned)
}

fn is_space(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\n' | '\r') || get_general_category(c) == Cat::SpaceSeparator
}

fn is_punctuation(c: char) -> bool {
    matches!(c as u32, 33..=47 | 58..=64 | 91..=96 | 123..=126)
        || matches!(
            get_general_category(c),
            Cat::ConnectorPunctuation
                | Cat::DashPunctuation
                | Cat::OpenPunctuation
                | Cat::ClosePunctuation
                | Cat::InitialPunctuation
                | Cat::FinalPunctuation
                | Cat::OtherPunctuation
        )
}

// ------------------------------------------------------- what the model cannot read

/// Pieces of at most `PIECE_CHARS`, cut after a sentence where there is one.
fn pieces(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for sentence in split_sentences(text.trim()) {
        match out.last_mut() {
            Some(last) if last.chars().count() + sentence.chars().count() < PIECE_CHARS => {
                last.push(' ');
                last.push_str(&sentence);
            }
            _ => out.push(sentence),
        }
    }
    out
}

/// The model knows Russian letters only and skips everything else in silence: numbers
/// are written out in words here and Latin letters are turned into Russian ones.
/// This is not part of Silero; it is rough, and numbers stay in the nominative.
pub fn spell_out(text: &str) -> String {
    const LATIN: [&str; 26] =
        ["а", "б", "к", "д", "е", "ф", "г", "х", "и", "дж", "к", "л", "м", "н", "о", "п", "к", "р", "с", "т", "у", "в", "в", "кс", "й", "з"];
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::new();
    let mut at = 0;
    while at < chars.len() {
        let c = chars[at];
        if c.is_ascii_digit() {
            let start = at;
            while at < chars.len() && chars[at].is_ascii_digit() {
                at += 1;
            }
            let digits: String = chars[start..at].iter().collect();
            match digits.parse::<u64>() {
                Ok(number) if digits.len() <= 9 => out.push_str(&number_words(number)),
                // Too long to be a quantity: read digit by digit.
                _ => out.push_str(&digits.chars().map(|d| number_words(d as u64 - '0' as u64)).collect::<Vec<_>>().join(" ")),
            }
            out.push(' ');
            continue;
        }
        if c.is_ascii_alphabetic() {
            let letter = LATIN[(c.to_ascii_lowercase() as u8 - b'a') as usize];
            if c.is_ascii_uppercase() {
                out.extend(letter.chars().enumerate().flat_map(|(i, l)| if i == 0 { l.to_uppercase().collect::<Vec<_>>() } else { vec![l] }));
            } else {
                out.push_str(letter);
            }
        } else {
            out.push(c);
        }
        at += 1;
    }
    out
}

fn number_words(number: u64) -> String {
    const ONES: [&str; 20] = [
        "ноль", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять", "десять", "одиннадцать", "двенадцать",
        "тринадцать", "четырнадцать", "пятнадцать", "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать",
    ];
    const TENS: [&str; 10] = ["", "", "двадцать", "тридцать", "сорок", "пятьдесят", "шестьдесят", "семьдесят", "восемьдесят", "девяносто"];
    const HUNDREDS: [&str; 10] = ["", "сто", "двести", "триста", "четыреста", "пятьсот", "шестьсот", "семьсот", "восемьсот", "девятьсот"];
    fn below_thousand(n: u64, feminine: bool, out: &mut Vec<String>) {
        out.extend((n >= 100).then(|| HUNDREDS[(n / 100) as usize].to_string()));
        let rest = n % 100;
        let (tens, ones) = if rest < 20 { (0, rest) } else { (rest / 10, rest % 10) };
        out.extend((tens > 0).then(|| TENS[tens as usize].to_string()));
        match ones {
            0 => {}
            1 if feminine => out.push("одна".into()),
            2 if feminine => out.push("две".into()),
            n => out.push(ONES[n as usize].to_string()),
        }
    }
    /// "тысяча", "тысячи" or "тысяч" after `n`.
    fn form<'a>(n: u64, forms: [&'a str; 3]) -> &'a str {
        match (n % 100, n % 10) {
            (11..=14, _) => forms[2],
            (_, 1) => forms[0],
            (_, 2..=4) => forms[1],
            _ => forms[2],
        }
    }
    if number == 0 {
        return ONES[0].into();
    }
    let mut out = Vec::new();
    let (millions, thousands, rest) = (number / 1_000_000, number / 1000 % 1000, number % 1000);
    if millions > 0 {
        below_thousand(millions, false, &mut out);
        out.push(form(millions, ["миллион", "миллиона", "миллионов"]).into());
    }
    if thousands > 0 {
        below_thousand(thousands, true, &mut out);
        out.push(form(thousands, ["тысяча", "тысячи", "тысяч"]).into());
    }
    below_thousand(rest, false, &mut out);
    out.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers_and_latin_are_spelt_out() {
        assert_eq!(number_words(1432), "одна тысяча четыреста тридцать два");
        assert_eq!(number_words(21_000), "двадцать одна тысяча");
        assert_eq!(number_words(2_000_015), "два миллиона пятнадцать");
        assert_eq!(spell_out("В 3 часа Arthur ушёл"), "В три  часа Артхур ушёл");
    }

    #[test]
    fn text_is_cut_like_python_does() {
        assert_eq!(split_keeping("кто-то, я", |c| c.is_whitespace() || WORD_BREAKS.contains(c)), ["кто-то", ", ", "я"]);
        assert_eq!(split_sentences("Да. Нет?  Ну!"), ["Да.", "Нет?", "Ну!"]);
        assert!(has_question_word("ну а кто это"));
        assert!(!has_question_word("это правда"));
    }

    /// With the exported model in `SILERO_DIR`: the stresses and the length of the sound
    /// have to be what Python gave for the same phrases (`reference.json`).
    #[test]
    fn the_port_agrees_with_python() {
        let Ok(dir) = std::env::var("SILERO_DIR") else { return };
        let dir = Path::new(&dir);
        tch::set_num_threads(1);
        let silero = Silero::load(dir).unwrap();
        #[derive(Deserialize)]
        struct Phrase {
            text: String,
            clean: String,
            accented: String,
            samples: usize,
        }
        let phrases: Vec<Phrase> = serde_json::from_str(&std::fs::read_to_string(dir.join("reference.json")).unwrap()).unwrap();
        let mut failed = 0;
        for phrase in phrases {
            let clean = silero.clean(&phrase.text);
            let accented = silero.accent(&clean).unwrap();
            let started = std::time::Instant::now();
            let (samples, _) = silero.speak_piece(&phrase.text, 4, 1.0).unwrap().unwrap();
            let ok = clean == phrase.clean && accented == phrase.accented && samples.len() == phrase.samples;
            failed += usize::from(!ok);
            println!(
                "{} {accented}\n   {} samples (python {}), {:.2} s",
                if ok { "ok  " } else { "DIFF" },
                samples.len(),
                phrase.samples,
                started.elapsed().as_secs_f32()
            );
            if accented != phrase.accented {
                println!("   python: {}", phrase.accented);
            }
        }
        assert_eq!(failed, 0);
    }
}
