"""Export validated LibriTTS_R metadata without modifying source audio."""
import argparse
import json
from pathlib import Path
import wave

from tts_pipeline.libritts_data import LibriTTSRawDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/data/db/LibriTTS_R')
    parser.add_argument('--output', default='data/librispeech_data.json')
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path = output.with_suffix('.report.json')
    if output.exists() or report_path.exists():
        raise FileExistsError('Output already exists; choose a new --output path')
    records, rejected, counts = [], [], {}
    seen = set()
    for split in ('train-clean-100', 'train-clean-360'):
        dataset = LibriTTSRawDataset(args.root, [split])
        count, seconds = 0, 0.0
        for index in range(len(dataset)):
            try:
                item = dataset[index]
                item['text'] = ' '.join(item['text'].split())
                if item['id'] in seen:
                    raise ValueError('Duplicate utterance ID')
                seen.add(item['id'])
                item['split'] = split
                records.append(item)
                count += 1
                seconds += item['duration_seconds']
            except (OSError, ValueError, EOFError, wave.Error, UnicodeError) as error:
                rejected.append({'audio': str(dataset.paths[index]), 'reason': str(error)})
        counts[split] = {'utterances': count, 'hours': seconds / 3600}
        print(split, counts[split], flush=True)
    with output.open('x', encoding='utf-8') as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    report = {'source': args.root, 'text_type': 'normalized', 'splits': counts,
              'accepted': len(records), 'rejected_count': len(rejected), 'rejected': rejected,
              'checks': 'Transcript readability/nonempty, WAV header, positive sample count, unique ID. No ASR or waveform quality filtering.'}
    with report_path.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(f'Saved {len(records)} records to {output}; rejected {len(rejected)}')


if __name__ == '__main__':
    main()
