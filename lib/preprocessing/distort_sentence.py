from lib.random.wrapper import KeyArray, bernoulli

def distort_sentence(sentence: str, key: KeyArray, keep_rate: float=0.8) -> str:
    '''
    TODO: Change to a more complicated distort method.
    '''

    words = sentence.split(' ')
    list_should_keep = bernoulli(key, p=keep_rate, shape=(len(words),))

    if list_should_keep.all().item():  # should keep all
        masked_words = ['<mask>', *words[1:]]  # guarantee that at least one word is masked
    else:
        masked_words = ['<mask>' if not should_keep else word for word, should_keep in zip(words, list_should_keep)]

    return ' '.join(masked_words)
