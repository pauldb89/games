import random


def load_words(path: str) -> list[str]:
    words = []
    with open(path, "r") as f:
        for line in f:
            words.append(line.strip())
    random.Random(0).shuffle(words)
    return words



def letter_index(letter: str) -> int:
    return ord(letter) - ord('a')


def get_letter(letter_id: int) -> str:
    assert 0 <= letter_id < 26
    return chr(ord('a') + letter_id)


class Vocab:
    def __init__(self, words: list[str], max_secret_options: int | None, skip_mask: bool = False) -> None:
        self.words = words
        self.max_secret_options = max_secret_options
        self.skip_mask = skip_mask
        self.mask_cache = {}

    def pick_secret(self, seed: int) -> str:
        return random.Random(seed).choice(self.words[:self.max_secret_options])

    def build_mask(self, prefix: str) -> list[int]:
        mask = [False] * 26
        for word in self.words:
            if word.startswith(prefix):
                mask[letter_index(word[len(prefix)])] = True
        return mask


    def get_mask(self, prefix: str) -> list[int]:
        if self.skip_mask:
            return [True] * 26

        if mask := self.mask_cache.get(prefix):
            return mask

        mask = self.build_mask(prefix)
        self.mask_cache[prefix] = mask
        return mask