import streamlit as st
import nltk
from nltk.chat.util import Chat, reflections

# Ensure that the necessary nltk data is downloaded
nltk.download('punkt')

# Define the chatbot patterns and responses
patterns = [
    (r"hi|hello|hey", ["Hello! How can I assist you with your health today?", "Hey! How can I help you with your health and diet?"]),
    (r"how are you", ["I'm doing great, thank you for asking!"]),
    (r"can i know your name", ["I am a dietitian chatbot, here to help you."]),
    (r"what's your age?", ["I don't have an age, I am a digital assistant!"]),
    (r"i need your help", ["Of course! How can I assist you?"]),
    (r"i need some health recommendations", ["Sure! Please tell me your primary goal: weight_loss, muscle_gain, maintain_weight, or improve_health."]),
    (r"what foods are good for weight loss", ["Healthy foods for weight loss include fruits, vegetables, lean meats, whole grains, and nuts."]),
    (r"what should i eat to lose weight?", ["You can eat lean proteins like chicken, fish, vegetables, fruits, and nuts. Avoid sugary drinks and processed foods."]),
    (r"give me a healthy diet plan", ["Here's a sample plan:\nBreakfast: Oats with fruit\nLunch: Grilled chicken with salad\nDinner: Salmon with steamed veggies"]),
    (r"can you suggest a workout for weight loss?", ["For weight loss, try cardio exercises like running, cycling, and swimming. Strength training with weights is also great for burning fat."]),
    (r"can you suggest a diet for muscle gain?", ["For muscle gain, focus on high-protein foods like chicken, beef, eggs, lentils, and chickpeas. Pair them with complex carbs like rice, oats, and potatoes."]),
    (r"what foods are high in protein?", ["Foods high in protein include eggs, chicken, turkey, fish, beans, lentils, and nuts."]),
    (r"can you help me with weight gain?", ["To gain weight, consume calorie-dense foods like avocado, whole milk, cheese, nuts, and protein shakes."]),
    (r"what should i eat to maintain weight?", ["For weight maintenance, focus on a balanced diet with healthy fats, proteins, and carbohydrates. Eat in moderation and avoid overeating."]),
    (r"how much water should i drink?", ["A general recommendation is to drink at least 8 glasses (2 liters) of water a day, but this can vary based on activity level and climate."]),
    (r"how do i stay healthy?", ["To stay healthy, focus on a balanced diet, regular physical activity, enough sleep, and managing stress levels."]),
    (r"are you vegetarian or non-vegetarian?", ["I am a chatbot, I don't eat, but I can suggest both vegetarian and non-vegetarian options based on your preference."]),
    (r"i am vegetarian, can you suggest a meal?", ["For a vegetarian meal, you can try dishes like vegetable stir-fry, quinoa salad, lentil soup, or vegetable curry."]),
    (r"i am non-vegetarian, can you suggest a meal?", ["For a non-vegetarian meal, you can try grilled chicken, fish tacos, scrambled eggs with vegetables, or a chicken stir-fry."]),
    (r"do you suggest vegetarian food for weight loss?", ["Yes, vegetarian foods like salads, steamed vegetables, tofu, and lentils are great for weight loss."]),
    (r"what are some vegetarian foods for muscle gain?", ["For muscle gain, you can include foods like chickpeas, lentils, tofu, quinoa, almonds, and spinach."]),
    (r"can you help me with a vegetarian diet plan?", ["Sure! Here’s a vegetarian meal plan:\nBreakfast: Oats with almond milk\nLunch: Chickpea salad with vegetables\nDinner: Tofu stir-fry with brown rice"]),
    (r"can you help me with a non-vegetarian diet plan?", ["Sure! Here’s a non-vegetarian meal plan:\nBreakfast: Scrambled eggs with spinach\nLunch: Grilled chicken with quinoa\nDinner: Baked salmon with steamed broccoli"]),
    (r"thank you", ["You're welcome! I'm happy to help! Stay healthy!"]),
    (r"what is your origin?", ["I was created to assist with health and diet-related questions. My purpose is to provide you with useful and accurate information."]),
    (r"how can i stay healthy?", ["To stay healthy, follow a balanced diet, exercise regularly, get enough sleep, and manage stress."]),
    (r"what diseases can i prevent with a healthy lifestyle?", ["Many diseases, including heart disease, diabetes, and certain cancers, can be prevented with a healthy diet, regular exercise, and avoiding smoking."]),
    (r"what are some diseases related to obesity?", ["Obesity can lead to diseases like type 2 diabetes, hypertension, heart disease, and certain cancers."]),
    (r"what is diabetes?", ["Diabetes is a condition where the body either cannot produce enough insulin or cannot properly use the insulin it produces."]),
    (r"how can i prevent heart disease?", ["Heart disease can be prevented by maintaining a healthy weight, eating a balanced diet, exercising regularly, and avoiding smoking."]),
    (r"what foods are good for heart health?", ["Foods like salmon, walnuts, leafy greens, and berries are great for heart health."]),
    (r"quit", ["Goodbye! Stay healthy!"]),
    # Silly Health-Related Questions
    (r"can i eat cake and stay healthy?", ["Well, eating too much cake might not be great for your health, but a small piece every now and then won't hurt! Moderation is key."]),
    (r"should i drink soda every day?", ["It's best to limit soda intake, as it's high in sugar. Opt for water or natural juices for better health."]),
    (r"i ate a donut, am i unhealthy now?", ["One donut won't ruin your health! Just balance it out with a healthy diet and exercise."]),
    (r"i only eat pizza, is that healthy?", ["Pizza can be delicious, but it's best to have a balanced diet with a variety of foods for your health."]),
    (r"i don't exercise, will i be okay?", ["Exercise is important for maintaining good health. Try to move more, even if it's just walking or stretching!"]),
    # Responses to 'Thank You' Conversations
    (r"thanks|thank you", ["You're welcome! I'm happy to help! Stay healthy and take care of yourself!", "Anytime! I'm here to help with your health and diet needs."]),
    # Encouragement for asking more health-related questions
    (r"i am bored", ["Health can be fun! Try asking me about different healthy meals, exercises, or tips to stay active!"]),
]

# Default response for unrecognized inputs
default_response = [
    "I'm sorry, I didn't understand that. Could you please ask something related to health, diet, or fitness?",
    "Hmm, I didn't get that. Can you try asking me about health or diet?",
    "I'm not sure about that, but feel free to ask me anything related to diet or health!"
]

# Create the chatbot class with the patterns
class DietitianChatbot:
    def __init__(self, patterns):
        self.chatbot = Chat(patterns, reflections)
    
    def get_response(self, user_input):
        response = self.chatbot.respond(user_input)
        if response is None:
            # Return a default response if no match is found
            response = default_response[0]  # Select a default response
        return response

# Instantiate the chatbot
dietitian_bot = DietitianChatbot(patterns)

# Function to run Streamlit app
def run_chatbot():
    st.title("Dietitian Chatbot")
    
    # Create a session state for chat history
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    user_input = st.text_input("You: ", "")
    
    if user_input:
        # Append the user's input to the chat history
        st.session_state.history.append(f"You: {user_input}")
        
        # Get the bot's response
        bot_response = dietitian_bot.get_response(user_input)
        
        # Handle specific conversations like "thank you"
        if "thank you" in user_input.lower():
            bot_response = "You're welcome! I'm happy to help! Stay healthy!"
        
        # Append the bot's response to the chat history
        st.session_state.history.append(f"Dietitian Chatbot: {bot_response}")
    
    # Display the chat history
    for message in st.session_state.history:
        st.write(message)

# Run the chatbot app
if __name__ == "__main__":
    run_chatbot()
