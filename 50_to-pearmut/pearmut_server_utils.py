# borrowed from pearmut/server/utils.py
# extracted only the part that we actually need, to reduce dependencies
# requires pip install wonderwords if the function is used

import random

_BAN_WORDS= {
    "abuse","beheading","bombing","cancer","cannibal","genocide","erection","fisting","floozie","homicide","intercourse",
    "kill","killing","misogyny","murder","racism","racist","rape","slave","slavery","suicide","terrorist","trafficker","x-rated"
}
def generate_user_name(existing_ids=None, rng=None):
    import wonderwords
    rng = rng or random.Random()
    rword = wonderwords.RandomWord(rng=rng)
    existing_ids = set(existing_ids) if existing_ids else set()
    while True:
        word_adjective = rword.random_words(amount=1, include_parts_of_speech=['adjective'])[0].lower()
        word_noun = rword.random_words(amount=1, include_parts_of_speech=['noun'])[0].lower()
        if word_adjective in _BAN_WORDS or word_noun in _BAN_WORDS:
            continue
        new_id = f"{word_adjective}-{word_noun}-{rng.randint(0, 999):03d}"
        if new_id not in existing_ids:
            return new_id
