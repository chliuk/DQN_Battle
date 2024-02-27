from transformers import pipeline

summarizer = pipeline("summarization")
res = summarizer("Trainers, We’ve made an update to how you log in to Pokémon GO using your Pokémon Trainer Club (PTC) account. For more detailed information about this change, please see this blog post. From now until January 31, 2024, local time, Trainers will receive Timed Research that rewards a Super Incubator and 1,000 Stardust when they validate or link their PTC account. Trainers who have already connected their PTC accounts to Pokémon GO will also receive this research and its rewards. Please note that Trainers who log in via a Niantic Kids account aren’t eligible to receive this Timed Research or these rewards. Trainers who previously used Niantic Kids but are older than their region’s age of digital consent are eligible.", min_length=5, max_length=100)

print(res)
