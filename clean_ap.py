import glob
import os
import click
import textgrid as tg
from tqdm import tqdm


def read_lab_file(lab_file):
    with open(lab_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    segments = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 3:
            raise ValueError(f"Invalid lab file format in {lab_file}")
        start = int(parts[0]) / 1e7  # convert from 100ns to seconds
        end = int(parts[1]) / 1e7  # convert from 100ns to seconds
        phoneme = parts[2]
        segments.append((start, end, phoneme))
    return segments


@click.command(help='')
@click.option('--label_dir', required=True, metavar='DIR', help='Label files (TextGrid or HTK lab)')
@click.option('--clean_label_dir', required=True, metavar='DIR', help='Clean label output dir')
@click.option('--phonemes', required=False, default="AP,SP,", type=str, help='Clean phonemes')
def clean(label_dir, clean_label_dir, phonemes):
    assert label_dir is not None, "Label directory (label_dir) cannot be None"
    assert clean_label_dir is not None, "Clean label output directory (clean_label_dir) cannot be None"
    assert label_dir != clean_label_dir, (
        "Label directory (label_dir) and Clean label output directory (clean_label_dir) cannot be the same")

    clean_phonemes = phonemes.split(',')

    label_files = glob.glob(f'{label_dir}/*')
    for label_file in tqdm(label_files):
        filename = os.path.basename(label_file)
        filename, ext = os.path.splitext(filename)
        out_label_path = os.path.join(clean_label_dir, filename + ext)

        if ext.lower() == '.textgrid':
            textgrid = tg.TextGrid()
            textgrid.read(label_file)
            words = textgrid[0]
            phones = textgrid[1]

            words_dict = {}
            word_cursor = 0
            for interval in words:
                if interval.minTime > word_cursor:
                    pass
                words_dict[interval.minTime] = {"start": interval.minTime, "end": interval.maxTime,
                                                "text": interval.mark, "phones": []}
                word_cursor = interval.maxTime

            for interval in words:
                word_start = interval.minTime
                word_end = interval.maxTime

                for phone in phones:
                    if phone.minTime >= word_start and phone.maxTime <= word_end:
                        words_dict[interval.minTime]["phones"].append({"start": phone.minTime, "end": phone.maxTime,
                                                                       "text": phone.mark})

            out = []
            for k, v in words_dict.items():
                if v['text'] not in clean_phonemes:
                    out.append(v)

            out_tg = tg.TextGrid()
            tier_words = tg.IntervalTier(name="words", minTime=0, maxTime=word_cursor)
            tier_phones = tg.IntervalTier(name="phones", minTime=0, maxTime=word_cursor)

            for item in out:
                tier_words.add(minTime=item['start'], maxTime=item['end'], mark=item['text'])
                if item['text'] == "SP" or item['text'] == "AP":
                    tier_phones.add(minTime=item['start'], maxTime=item['end'], mark=item['text'])
                    continue
                for phone in item['phones']:
                    tier_phones.add(minTime=phone['start'], maxTime=phone['end'], mark=phone['text'])

            out_tg.tiers.insert(0, tier_words)
            out_tg.tiers.insert(1, tier_phones)
            out_tg.write(out_label_path)
        
        elif ext.lower() == '.lab':
            try:
                segments = read_lab_file(label_file)
                with open(out_label_path, 'w', encoding='utf-8') as file:
                    for start, end, phoneme in segments:
                        if phoneme not in clean_phonemes:
                            file.write(f"{int(start * 1e7)} {int(end * 1e7)} {phoneme}\n")
            except ValueError as e:
                print(e)
        else:
            print(f"Unsupported file extension {ext} for file {label_file}")


if __name__ == '__main__':
    clean()
