from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

texts = [
    'This is the first text.',
    'This is the second text.',
    'This is the third text.',
    'This is the fourth text.',
    'This is the fifth text.',
    'This is the sixth text.',
    'This is the seventh text.',
    'This is the eighth text.',
    'This is the ninth text.',
    'This is the tenth text.',
    'This is the eleventh text.',
    'This is the twelfth text.',
    'This is the thirteenth text.',
    'This is the fourteenth text.',
    'This is the fifteenth text.',
    'This is the sixteenth text.',
    'This is the seventeenth text.',
    'This is the eighteenth text.',
    'This is the nineteenth text.',
    'This is the twentieth text.'
]
tokenizer.pad_token = tokenizer.eos_token
encoded_batch = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512, add_special_tokens=True, return_attention_mask=True, return_token_type_ids=False)

for i in range(0, len(texts), 32):
    batch = encoded_batch['input_ids'][i:i+32]
    attention_mask = encoded_batch['attention_mask'][i:i+32]
    print('Batch {}:'.format(i // 32))
    print(batch)
    print(attention_mask)
