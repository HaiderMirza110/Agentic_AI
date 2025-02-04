from crewai.flow.flow import start, Flow, listen
from litellm import completion
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("API_KEY")

class PromptChaining(Flow):
    @start()
    def sportman(self):
        result = completion(
            model="gemini/gemini-1.5-flash",
            api_key=API_KEY,
            messages=[{"content": "Return top 5 Crickter in the world of current era", "role": "user"}]
        )
        output = result["choices"][0]["message"]["content"]
        return output

    @listen(sportman)
    def stats(self, sportman_output):
        result = completion(
            model="gemini/gemini-1.5-flash",
            api_key=API_KEY,
            messages=[{"content": f"Write some fun facts each of them about {sportman_output}'s career", "role": "user"}]
        )
        output1 = result["choices"][0]["message"]["content"]
        return output1

    @listen(stats)
    def save_output(self, stats_output):
        with open("stats.md", "w") as file:
            file.write(stats_output)
        return stats_output


def career():
    chain = PromptChaining()
    output = chain.kickoff()
    print(f"Final result is: {output}")
