def letter_index(letter: str) -> int:
    return ord(letter) - ord('a')


def get_letter(letter_id: int) -> str:
    assert 0 <= letter_id < 26
    return chr(ord('a') + letter_id)


class Vocab:
    def __init__(self, path: str | None = None, words: list[str] | None = None, skip_mask: bool = False) -> None:
        assert (words is None) != (path is None), "Exactly one of path or words must be defined"

        if words is not None:
            self.words = words
        else:
            self.words = []
            with open(path, "r") as f:
                for line in f:
                    self.words.append(line.strip())
        self.skip_mask = skip_mask
        self.mask_cache = {}

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