"""One-time export of Silero TTS v5 (v5_5_ru) into files a Rust program can load with
libtorch, without Python: three TorchScript modules and the dictionaries around them.

    python export_silero.py <v5_5_ru.pt> <output folder>

Needs only PyTorch. The model is https://models.silero.ai/models/tts/ru/v5_5_ru.pt
(CC BY-NC 4.0: non-commercial use only).
"""
import json
import os
import sys
import warnings

import torch

warnings.filterwarnings("ignore")
package, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)

model = torch.package.PackageImporter(package).load_pickle("tts_models", "model")
part = model.packages[0]
stress = part.accentor
# The homograph model ships with 8-bit word embeddings; this restores them, as the
# first `apply_tts` call would.
part.unpack_q_model()
part.q_model_unpacked = True

torch.jit.save(part.models[0], os.path.join(out, "tts.jit"))
torch.jit.save(stress.accentor.model, os.path.join(out, "accentor.jit"))
torch.jit.save(stress.homosolver.model, os.path.join(out, "homosolver.jit"))

tokenizer = stress.homosolver.tokenizer
data = {
    "symbols": part.symbol_to_id,
    "speakers": part.speaker_to_ids[0],
    # word -> [where the stress mark goes, which letter becomes "ё" or -1]
    "exceptions": stress.accentor.exceptions,
    # word -> its readings with "+" before the stressed vowel
    "homographs": stress.homosolver.homodict,
    "vocab": tokenizer.vocab,
    "never_split": sorted(tokenizer.never_split),
    "tokens": {
        "unk": tokenizer.unk_token_id,
        "cls": tokenizer.cls_token_id,
        "sep": tokenizer.sep_token_id,
        "pad": tokenizer.pad_token_id,
        "homo_start": tokenizer.homo_start_id,
        "homo_end": tokenizer.homo_end_id,
    },
}
with open(os.path.join(out, "data.json"), "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False)

# What Python makes of a few phrases, for the Rust port to be checked against.
phrases = [
    "Колокол Рассвета спит под часовней. Три удара на закате выманивают Змея.",
    "Я открыл замок в старом замке. Все знают, что все кончено.",
    "Кто-то из-за угла сказал: что это? Разве не так, а?",
    "Ёжик нёс мёд, но берег его не берёг.",
    "Сорок сорок съели сорок пирогов — и что-то ещё!",
    "Стоит ли дорога того, чтобы стоять на большой дороге?",
]
reference = []
torch.set_num_threads(1)
torch.manual_seed(0)
for text in phrases:
    clean, _, _ = part.prepare_text_input(text.strip(), None)
    accented = stress(clean)
    audio = model.apply_tts(text=text, speaker="xenia", sample_rate=24000)
    reference.append({"text": text, "clean": clean, "accented": accented, "samples": int(audio.shape[0])})
with open(os.path.join(out, "reference.json"), "w", encoding="utf-8") as f:
    json.dump(reference, f, ensure_ascii=False, indent=1)
print("exported to", out)
