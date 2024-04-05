import pandas as pd

# Assuming df is a DataFrame with your stock data
df = pd.DataFrame({
    'Date': ['2024-03-01', '2024-03-02', '2024-03-03'],  # Example dates
    'Open': [100, 102, 103],
    'Close': [105, 101, 104],
    'High': [106, 103, 105],
    'Low': [99, 100, 102],
})

# Simple function to determine if a day's candlestick is bullish or bearish
def get_candle_type(row):
    if row['Close'] > row['Open']:
        return 'bullish'
    elif row['Close'] < row['Open']:
        return 'bearish'
    else:
        return 'neutral'

# Add a column to df with the candle type
df['Candle_Type'] = df.apply(get_candle_type, axis=1)

# Now create a text description of the patterns you've identified
patterns_text = ' '.join([f"On {row['Date']}, there was a {row['Candle_Type']} candle with a high of {row['High']} and a low of {row['Low']}." for index, row in df.iterrows()])

# Here is the text that we will pass to the GPT model
print(patterns_text)

# You would then use this text as the prompt for your GPT API call
# as shown in the previous example

# Assume 'df' is a DataFrame with stock data containing 'Open', 'High', 'Low', 'Close', 'Volume'.

# Construct a summary description of the stock's recent performance
summary = f"The stock had a closing price range from ${df['Close'].min()} to ${df['Close'].max()} " \
          f"over the last {len(df)} trading days. The average trading volume was {df['Volume'].mean():,.0f}. "

# You might want to include more sophisticated analysis to detect trends, patterns, and other points of interest.
# For this example, let's say you detect a significant increase:
if df['Close'].iloc[-1] > df['Close'].iloc[0]:
    summary += "The closing price has increased overall during this period. "

# If there was a day with a particularly high volume, note that too:
if df['Volume'].max() > 1.5 * df['Volume'].mean():
    high_volume_day = df['Volume'].idxmax()
    summary += f"The highest trading volume was on {high_volume_day.strftime('%Y-%m-%d')}."

# Now 'summary' contains a text description of the stock's performance. Present this to ChatGPT.
print(summary)

# You would then take this summary and present it to ChatGPT for analysis.
# Since I cannot initiate the interaction with ChatGPT, you would do it in your application.
import openai

openai.api_key = 'your-api-key'  # Replace with your actual API key from OpenAI.

# Here's the summary we created earlier.
summary = """
The stock had a closing price range from $135 to $150 over the last 30 trading days.
The average trading volume was 2,500,000. The closing price has increased overall during this period.
The highest trading volume was on 2024-04-02.
"""

# Prepare the prompt for GPT.
# Here, you can customize the prompt to ask specific questions or to guide the model on the analysis you're expecting.
prompt = f"Given the following summary of a stock's performance, provide an analysis and potential reasons for the trends:\n\n{summary}"

response = openai.Completion.create(
  engine="text-davinci-003",  # Or another model name if you prefer
  prompt=prompt,
  max_tokens=150,  # You can adjust the number of tokens (word count) as needed
)

print(response.choices[0].text.strip())
