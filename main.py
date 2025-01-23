import discord
from discord.ext import commands
from ultralytics import YOLO
import torch
import os
import random
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load the YOLO Model
try:
    yolo_model = YOLO("best.pt")
except Exception as e:
    raise RuntimeError(f"Failed to load YOLO model: {e}")

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Item Categories Mapping
ITEM_CATEGORIES = {
    'Sulfur_stack': 'Sulfur',
    'gunpowder': 'Gunpowder',
    'explosives': 'Explosives',
    'cooked_sulfur': 'Cooked Sulfur',
    'pipes': 'Pipes',
    'AK47': 'AK47',
    'Metal_ore': 'Metal Ore',
    'Diesel': 'Diesel',
    'High_quality_metal': 'High-Quality Metal',
    'Crude_oil': 'Crude Oil',
    'Cloth': 'Cloth',
    'Scrap': 'Scrap',
    'HQM_ore': 'HQM Ore',
    'Rocket': 'Rocket',
    'c4': 'C4',
    'charcoal': 'Charcoal',
    'MLRS': 'MLRS',
    'MLRS_module': 'MLRS Module',
    'Metal_fragments': 'Metal Fragments',
    'Low_grade_fuel': 'Low Grade Fuel'
}

# Discord Bot Setup
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# Function to process images and count detections
def process_image(image_path):
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file {image_path} does not exist.")

        print(f"Processing image: {image_path}")

        # Perform YOLO inference
        results = yolo_model(image_path)
        if not results or not hasattr(results[0], "boxes") or not hasattr(results[0].boxes, "data"):
            raise ValueError("YOLO results do not contain valid bounding box data.")

        detections = results[0].boxes.data.cpu().numpy()
        counts = {}

        for detection in detections:
            try:
                x1, y1, x2, y2, conf, class_id = detection

                if int(class_id) not in results[0].names:
                    print(f"Invalid class_id {class_id} detected. Skipping.")
                    continue

                class_name = results[0].names[int(class_id)]
                if class_name in ITEM_CATEGORIES:
                    item_name = ITEM_CATEGORIES[class_name]
                    counts[item_name] = counts.get(item_name, 0) + 1
            except Exception as e:
                print(f"Error processing detection: {detection}, error: {e}")
                continue

        print(f"Final inventory counts: {counts}")
        return counts
    except Exception as e:
        print(f"Error processing image: {e}")
        raise

# Function to generate a dynamic response
def generate_response(user_name, item_counts):
    try:
        sorted_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)

        if not sorted_items:
            return f"Sorry, {user_name}, I couldn't find anything in that image!"

        # Get top 1-5 items
        num_items = random.randint(1, min(5, len(sorted_items)))
        top_items = [item[0] for item in sorted_items[:num_items]]

        # Roll for response type
        print("Rolling for response type!")
        roll = random.randint(1, 100)
        print(f"{roll} was rolled!")

        if roll <= 20:  # Snarky (1-10)
            print("Initiating a snarky response.")
            all_items = set(ITEM_CATEGORIES.values())
            detected_items = set(item_counts.keys())
            missing_items = list(all_items - detected_items)
            least_item = sorted_items[-1][0] if sorted_items else None

            snarky_lines = []
            if missing_items:
                snarky_lines += [
                    f"Uh, {user_name}, where’s the {random.choice(missing_items)}? We kind of need that too!",
                    f"{user_name}, if you dont start bringing back {random.choice(missing_items)} you're gonna have to live outside!",
                    f"{user_name}, Das ist inakzeptabel, wir brauchen mehr {random.choice(missing_items)}!",
                    f"{user_name}, Im gonna tell your mom that you aren't bringing back enough {random.choice(missing_items)}!",
                ]
            if least_item:
                snarky_lines += [
                    f"But seriously, {user_name}, only a few {least_item}? Bold move!",
                ]
            return random.choice(snarky_lines) if snarky_lines else f"Come on, {user_name}, step it up!"

        elif 21 <= roll <= 97:  # Compliment (21-97)
            print("Initiating a compliment.")
            compliments = [
                f"Wow, {user_name}! You brought in so much {top_items[0]}!",
                f"Looks like {top_items[0]} is your specialty, {user_name}!",
                f"You’ve got {top_items[0]} for days, {user_name}!",
                f"Stocking up on {top_items[0]} like a pro, {user_name}!",
                f"Nice job bring back so much {top_items[0]}, you wont have to live outside this wipe, {user_name}!",
                f"Thats enough {top_items[0]} to fill a barrel, nice job {user_name}!",
                f"This guy brought back enough {top_items[0]} that we dont have to beat em! Well done, {user_name}!",
                f"Any more {top_items[0]} and we would be out of box space! Well done, {user_name}!",
                f"Make sure to put all of those {top_items[0]} in the right box! Good job {user_name}!",
                f"Wow, das ist eine Menge {top_items[0]}, gut gemacht, {user_name}!",
            ]
            if len(top_items) > 1:
                compliments += [
                    f"Wow, {user_name}, you’re really stocking up on {top_items[0]} and {top_items[1]}!",
                    f"{user_name}, you’re practically swimming in {top_items[0]} and {top_items[1]}!",
                ]
            if len(top_items) > 2:
                compliments += [
                    f"You’re killing it with {', '.join(top_items[:3])}, {user_name}!",
                    f"Incredible haul of {', '.join(top_items[:3])}, {user_name}!",
                ]
            return random.choice(compliments)

        elif 98 <= roll <= 99:  # Special message: Tater tots
            print("Initiating a special message: Tater tots.")
            return "I really like tater tots."

        elif roll == 100:  # Special message: Bunny
            print("Initiating a special message: Bunny.")
            return "I am a bunny!"

    except Exception as e:
        print(f"Error generating response: {e}")
        return f"Oops, something went wrong with your inventory analysis, {user_name}!"

# Bot Events
@bot.event
async def on_ready():
    print(f"Bot is ready. Logged in as {bot.user}")

@bot.event
async def on_message(message):
    if message.channel.name == "loot-brags" and message.attachments:
        for attachment in message.attachments:
            file_path = f"temp/{attachment.filename}"
            os.makedirs("temp", exist_ok=True)

            try:
                await attachment.save(file_path)
                counts = process_image(file_path)

                if counts:
                    response = generate_response(message.author.display_name, counts)
                    await message.channel.send(response)
                else:
                    await message.channel.send(f"Couldn't find anything notable, {message.author.display_name}! Maybe try again?")
            except Exception as e:
                await message.channel.send(f"Error processing the image: {e}")
            finally:
                if os.path.exists(file_path):
                    os.remove(file_path)

@bot.command()
async def inventory(ctx):
    if not ctx.message.attachments:
        await ctx.send("Please attach an image containing the inventory.")
        return

    for attachment in ctx.message.attachments:
        file_path = f"temp/{attachment.filename}"
        os.makedirs("temp", exist_ok=True)

        try:
            await attachment.save(file_path)
            counts = process_image(file_path)

            if counts:
                sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
                inventory_text = "\n".join([f"{item}: {count}" for item, count in sorted_items])
                await ctx.send(f"Detected inventory:\n{inventory_text}")
            else:
                await ctx.send("Couldn't detect anything in the image. Try again?")
        except Exception as e:
            await ctx.send(f"Error processing the image: {e}")
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

# Run the bot
TOKEN = os.getenv("DISCORD_TOKEN")
if not TOKEN:
    raise ValueError("Discord bot token not found in .env file")

bot.run(TOKEN)
