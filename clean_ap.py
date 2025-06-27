import glob
import os.path

import click
import textgrid as tg
from tqdm import tqdm


@click.command(help='')
@click.option('--tg_dir', required=True, metavar='DIR', help='Textgrid files')
@click.option('--clean_tg_dir', required=True, metavar='DIR', help='Clean textgrid output dir')
@click.option('--phonemes', required=False, default="AP,SP,", type=str, help='Clean phonemes')
def clean(tg_dir, clean_tg_dir, phonemes):
    assert tg_dir is not None, "TextGrid directory (tg_dir) cannot be None"
    assert clean_tg_dir is not None, "TextGrid output directory (tg_out_dir) cannot be None"
    assert tg_dir != clean_tg_dir, (
        "TextGrid directory (tg_dir) and Clean textgrid output directory (clean_tg_dir) cannot be the same")

    clean_phonemes = phonemes.split(',')

    tg_files = glob.glob(f'{tg_dir}/*.TextGrid')
    for tg_file in tqdm(tg_files):
        filename = os.path.basename(tg_file)
        filename, _ = os.path.splitext(filename)
        out_tg_path = os.path.join(clean_tg_dir, filename + '.TextGrid')

        textgrid = tg.TextGrid()
        textgrid.read(tg_file)
        words = textgrid[0]
        phones = textgrid[1]

        words_dict = {}
        word_cursor = 0
        for interval in words:
            if interval.minTime > word_cursor:
                # words_dict[word_cursor] = {"start": word_cursor, "end": interval.minTime, "text": "", "phones": []}
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
        out_tg.write(out_tg_path)


if __name__ == '__main__':
    clean()
