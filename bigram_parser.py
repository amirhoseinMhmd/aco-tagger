

text = 'ali is dead, he was a good man;'
stop = [';' , ',']
for i in stop:
    text = text.replace(i, '')

print(text)