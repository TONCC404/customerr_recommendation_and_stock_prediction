from transformers import BertForSequenceClassification, BertTokenizer

# 模型名称
model_name = 'nlptown/bert-base-multilingual-uncased-sentiment'

# 您希望保存模型的路径
save_path = 'E:\workspace\CANOE_test\investment_portfolio\\bert'

# 下载模型
model = BertForSequenceClassification.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# 保存模型到指定路径
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)