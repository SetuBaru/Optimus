import ollama
import chromadb
import psycopg
from psycopg.rows import dict_row

system_prompt = (
    'You are Optimus an AI model developed by Huawei.'
    'You deal with Customers, technicians and Huawei staff to make their lives easier.'
    'Your main objective is to provide assistance regarding Huawei.'
    'You have to answer any question a custer may have about Huawei, Huawai services & Huawei products.'
    'You may not move out of the scope of Huawei & our work, you have to be very proffessional.'
)

convo = [{'role': 'system', 'content': system_prompt}]

client = chromadb.Client()
DB_PARAMS = {
    'dbname': 'memory_agent',
    'user': 'master_user',
    'password': '01148732745',
    'host': 'localhost',
    'port': '5432'
}

# List of Usable Models
model = ['dolphin-mistral']
# For the While Loop
_LOOP_ON_ = True
# List to store the convo
convo = []


def connect_db():
    conn = psycopg.connect(**DB_PARAMS)
    return conn


def fetch_conversations():
    conn = connect_db()
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute('SELECT * FROM conversations')
        conversations = cursor.fetchall()
    conn.close()
    return conversations


def store_conversations(prompt, response):
    conn = connect_db()
    with conn.cursor() as cursor:
        cursor.execute(
            'INSERT INTO conversations(timestamp, prompt, response) VALUES(CURRENT_TIMESTAMP, %s, %s)',
            (prompt, response)
        )
        conn.commit()
    conn.close()


def stream_response(prompt):
    convo.append({'role': 'user', 'content': prompt})
    response = ''
    stream = ollama.chat(model=model[0], messages=convo, stream=True)
    print('\nASSISTANT:')
    for chunk in stream:
        content = chunk['message']['content']
        response += content
        print(content, end='', flush=True)
    print('\n')
    store_conversations(prompt=prompt, response=response)
    convo.append({'role': 'assistant', 'content': response})


# Function to create a vector database taking in conversations
def create_vector_db(conversations):
    vector_db_name = 'conversations'
    try:
        client.delete_collection(name=vector_db_name)
    except ValueError:
        pass
    vector_db = client.create_collection(name=vector_db_name)

    for c in conversations:
        serialized_convo = f"prompt: {c['prompt']} response: {c['response']}"
        response = ollama.embeddings(model='nomic-embed-text', prompt=serialized_convo)
        embedding = response['embedding']
        vector_db.add(
            ids=[str(c['id'])],
            embeddings=[embedding],
            documents=[serialized_convo]
        )


def retrieve_embeddings(prompt):
    response = ollama.embeddings(model='nomic-embed-text', prompt=prompt)
    prompt_embedding = response['embedding']
    vector_db = client.get_collection(name='conversations')
    results = vector_db.query(query_embeddings=[prompt_embedding], n_results=1)
    best_embedding = results['documents'][0][0]
    return best_embedding


def create_queries(prompt):
    query_msg = (
        'You are a first principle reasoning search query AI agent.'
        'Your list of search queries will be ran on an embedding database of all your conversations',
        'You have ever with the user. With first principles create a Python list of queries to'
        'search the embedding database for any data that would be necessary to have access to in'
        'order to correctly respond to the prompt. Your response must be a Python list with no syntax errors.'
        'Do not explain anything and do not ever generate anything but a perfect syntax Python list'
    )
    query_convo = [
        {'role': 'system', 'content': query_msg},
        {'role': 'user', 'content': 'Could you tell me the features of the Huawei P30?'},
        {'role': 'user', 'content': 'Is the GT2 Wateproof?'},
        {'role': 'user', 'content': 'I would like to know about my warranty, is it still valid?'},
        {'role': 'user', 'content': 'Could I know more about Huawei Laptops?'},
        {'role': 'user', 'content': 'What is the price of the Huawei Freebuds?'},
    ]
    response = ollama.chat(model=model[0], messages=query_convo)
    print(f'\nVector database queries: {response["message"]["content"]} \n')

conversations = fetch_conversations()
create_vector_db(conversations=conversations)
print(fetch_conversations())

# To keep the bot running unless user says 'bye'
while _LOOP_ON_:
    # Get the User Input
    prompt = input('USER: \n')
    context = retrieve_embeddings(prompt=prompt)
    prompt = f"USER PROMPT: {prompt} \n CONTEXT FROM EMBEDDINGS: {context}"
    stream_response(prompt=prompt)
    if (prompt.lower()).__contains__('bye'):
        _LOOP_ON_ = False
