import discord
import asyncio
import time
from nltk.corpus import wordnet
from collections import defaultdict
import re
import requests
from model import *


def generate_image(keyword):
    url = 'https://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word='+keyword+'&ct=201326592&v=flip'
    result = requests.get(url)
    html = result.text
    pic_url = re.findall('"objURL":"(.*?)",',html,re.S)
    i = 0

    return pic_url[0]


PARTS_OF_SPEECH = {
    wordnet.NOUN: "Noun",
    wordnet.ADJ: "Adjective",
    wordnet.VERB: "Verb",
    wordnet.ADJ_SAT: "Adjective Satellite",
    wordnet.ADV: "Adverb"
}


def format_meaning(word, synsets):
    reply = f'**{word}**\n\n'

    # Group by POS
    grouped = defaultdict(list)
    for synset in synsets:
        grouped[PARTS_OF_SPEECH[synset.pos()]].append(synset.definition())

    for pos, definitions in grouped.items():
        reply += f'*{pos}*\n'
        for counter, definition in enumerate(definitions, 1):
            reply += f'    {counter}. {definition}\n'
    return reply


async def handle_message(message):

    word = message.content[5:]
    embed = discord.Embed(title=f"Our spies have sighted the {word} ")
    embed.set_image(url=generate_image(word))
    await message.channel.send(embed=embed)
    try:
        synsets = wordnet.synsets(word)
        if synsets:
            reply = format_meaning(word, synsets)
        else:
            reply = f'No extra information could be acquired.'
    except:
        reply = 'Sorry, an error occurred while fetching that definition.'
    await message.channel.send(reply)


messages = joined = 0

client = discord.Client()


async def update_stats():
    await client.wait_until_ready()
    global messages, joined

    while not client.is_closed():
        try:
            with open("stats.txt", "a") as f:
                f.write(f"Time: {int(time.time())}, Messages: {messages}, Members Joined: {joined}\n")

            messages = 0
            joined = 0

            await asyncio.sleep(5)
        except Exception as e:
            print(e)
            await asyncio.sleep(5)


@client.event
async def on_message(message):
    global messages
    messages += 1
    id = client.get_guild(709006557358063636)
    print(message.content) # Now every message sent will be printed to the console
    print("--------------------------------------------------------------------")

    channels = ["commands", "general"]
    users = ["dynamic#6160"]
    if str(message.channel) in channels and str(message.author) in users: # Places restrictions on responsivity of the bot

        if message.content.find("hello") or message.content.find("awaken") != -1:
            await message.channel.send("Ravel in my presence my brethren")  # If the user says !hello we will send back hi
        elif message.content == "!users" :
            await message.channel.send("## of Members: ", id.member_count)

    if message.content.startswith('/spy '):
        await handle_message(message)



    @client.event
    async def on_member_join(member):
        global joined
        joined += 1
        for channel in member.guild.channels:
            if str(channel) == "general":  # We check to make sure we are sending the message in the general channel
                await channel.send_message(f"""Welcome to the server {member.mention}""")


client.loop.create_task(update_stats())
client.run("NzA5MDAzMzQ5MjgxOTk2ODIx.Xu5zYQ.GCx4CFSiUGf7GdUEv2s3x9bhc80")

print(new_list[0])
gennames.generate()