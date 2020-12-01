import pynlpir

pynlpir.open()
s = '今天天气真是好呀'
segments = pynlpir.segment(s)
stop_cx =['modal particle', 'punctuation mark', 'noun of locality','particle','numeral']
for segment in segments:
    #if segment[1] not in stop_cx:
        print(segment[0], '\t', segment[1])
print('---')
key_words = pynlpir.get_key_words(s, weighted=True)
for key_word in key_words:
    print(key_word[0], '\t', key_word[1])
    
pynlpir.close()

'''
今天 	 time word
天气 	 noun
真是 	 adverb
好 	 adjective
呀 	 modal particle
---
今天 	 2.2
天气 	 2.0
'''
