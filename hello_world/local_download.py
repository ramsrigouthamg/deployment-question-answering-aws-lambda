from transformers import AutoTokenizer, AutoModelForQuestionAnswering

tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")


model.save_pretrained('./roberta-base-squad2')
tokenizer.save_pretrained('./roberta-base-squad2')