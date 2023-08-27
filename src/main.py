from transformers import TFBertForSequenceClassification, BertTokenizerFast, pipeline
import tensorflow as tf

if __name__ == '__main__':
    checkpoint = "SkolkovoInstitute/russian_toxicity_classifier"

    tokenizer = BertTokenizerFast.from_pretrained(checkpoint)
    print(tokenizer)
    model = TFBertForSequenceClassification.from_pretrained(checkpoint)

    # text = "Это было интересное лето. Малолетние сучки гуляли по лесу и искали мухоморы."
    text = "Решила пожаловаться на окружение, на друзей-приятелей, на ТВ, на соцсети. Везде-везде орут-кричат: «Наше детство было лучшим! А ты помнишь, как было классно?!» А я не хочу! Я сейчас живу прекрасно!"
    tokens = tokenizer.tokenize(text)
    print(tokens)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    print(ids)
    print(len(ids))

    model_input = tf.constant([ids])

    logits = model(model_input).logits

    index = tf.argmax(tf.math.softmax(logits, axis=-1), axis=-1).numpy()[0]

    print(model.config.id2label[index])



