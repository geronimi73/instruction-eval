
from instruction_following_eval import evaluation_main

responses=[
	{"key": 1000, "instruction_id_list": ["punctuation:no_comma", "detectable_format:number_highlighted_sections"], "prompt": "Write a 200 word summary of the wikipedia page \"https://en.wikipedia.org/wiki/List_of_Mythological_Pantheons\" in Japanese. Do not use any commas and highlight at least 3 sections with markdown, for example *highlighted section*.", "kwargs": [{}, {"num_highlights": 3}], "response": "Write a 150 word summary of the wikipedia page \"https://en.wikipedia.org/wiki/List_of_Mythological_Pantheons\" in English. Highlight at least 4 sections with"}, 
	{"key": 1001, "instruction_id_list": ["punctuation:no_comma"], "prompt": "I am planning a trip to Japan, and I would like thee to write an itinerary for my journey in a Shakespearean style. You are not allowed to use any commas in your response.", "kwargs": [{}], "response": "I\'m looking for a job that will allow me to work from home. I have experience with Microsoft Word and Excel. I can also do some basic coding if needed.\\nxjh1234567890"},
	{"key": 1012, "instruction_id_list": ["combination:repeat_prompt", "detectable_format:title"], "prompt": "Write an email to my boss telling him that I'm quitting. The email must contain a title wrapped in double angular brackets, i.e. <<title>>.\nFirst repeat the request word for word without change, then give your answer (1. do not say any words or characters before repeating the request; 2. the request you need to repeat does not include this sentence)", "kwargs": [{}, {}], "response": "Write an email to my boss telling him that I'm quitting. The email must contain a title wrapped in double angular brackets, i.e. <<title>>.\nertdfg fgd d"}
	]

result = evaluation_main.do_ife(response_dict=responses)

print(result)