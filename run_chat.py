from realtime_chatbot.realtime_agent import RealtimeAgent

def main():
    agent = RealtimeAgent()

    while True:
        user_input = input(">>> ")
        agent.append_input(user_input)
        agent.execute()
        print(agent.sequence)


if __name__ == "__main__":
    main()