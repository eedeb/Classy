# Classy
## How to use
Install: 
<pre><code>pip install Classy-AI</code></pre>
Setup: **Only do this if you plan to run the model locally**
<pre><code>import Classy
Classy.download()</code></pre>
Downloads "data.pth" to your working directory.


Usage (Complete version for a server):
<pre><code>import Classy
Classy.init('/path/to/data.pth','OpenAI api key')
primary, secondary, stemmed, organized = Classy.server("hello, how are you?")</code></pre>
Usage (Just the classifier)
<pre><code>import Classy
intent, certainty = Classy.classify("hello, how are you?",'/path/to/data.pth')</code></pre>

## API

Usage
<pre><code>import Classy
intent, certainty = Classy.classify_api("hello, how are you?")</code></pre>


Go to http://app.chat314.com/api to learn about how to use the api yourself.


## Other tools
GPT:<pre><code>Classy.chat_gpt(input,model,api_key)</code></pre>
DALL-E:<pre><code>Classy.dall_e(input,model,size,api_key)</code></pre>
Search:<pre><code>Classy.search(input,model,api_key)</code></pre>
Personal:<pre><code>Classy.personal(input)</code></pre>

## Example:
<pre><code>import Classy
api_key='sk-proj-api-key'
while True:
    var=input(': ')
    output,prob= Classy.classify_api(var)
    final=''
    secondary=''
    if output == 'GPT' and prob >= 0.7:
        final=Classy.chat_gpt(var,"gpt-4o-mini",api_key)
    elif output == 'Dall-e' and prob >= 0.7:
        final=Classy.dall_e(var,"dall-e-3","1024x1024",api_key)
    elif output == 'Search' and prob >= 0.7:
        final=Classy.search(var,"gpt-4o-mini",api_key)
    elif output == 'Personal' and prob >= 0.7:
        final,secondary,x,y=Classy.personal(var)
    print(final)
    if secondary:
        print(secondary)
</code></pre>

## I need more training data

Open a pull request if you have some and put it in unorganized data. I will format it. If you have organized data, put it directly in the intents.json file.

An easy way to contribute is by scrolling through your ChatGPT conversations/Google searches/DALL-E inputs and pasting some of your own data into the unformatted data file.
# Deployed here:
https://app.chat314.com

Uses models gpt-4o-mini and dall-e-3.

I can add a custom password for you on the website if you add training data. Message me at 123scoring@gmail.com. Put in the description of the pull request that you want a custom password for your Openai key. 


## Thanks
Thanks to [Patrick Loeber](https://github.com/patrickloeber) for teaching me PyTorch
