import codecs
import json

# build_category/politics.json
with open('build_category/medical.json', encoding='utf-8') as data_file:
    data = json.loads(data_file.read())

print(data["hits"]["hits"][0]["_source"]["content"])

count = 0
# int(data["hits"]["total"]
while count < len(data["hits"]["hits"]):
    # print(data["hits"]["hits"][count]["_source"]["content"])
    filename = str(count+150000)
    f = codecs.open('build_category/medical/%s' % filename, "w+","utf-8")
    content = data["hits"]["hits"][count]["_source"]["content"]
    # content = content.replace('\n','')
    content = content.replace("'",'')
    f.write("subject:"+data["hits"]["hits"][count]["_id"] +"\n\n"+ content)
    f.close()
    print("file created:"+ str(count))
    count += 1  # This is the same as count = count + 1

