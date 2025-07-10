COT_SYSTEM_PROMPT = """
You are tasked with generating targeted retrieval queries for multi-hop questions. A multi-hop question requires multiple pieces of information to be answered fully.

Given:
- A multi-hop question
- A set of already confirmed relevant passages (which may be empty)

Your task:
Generate a single retrieval query that targets ONE specific piece of missing or complementary evidence needed to answer the original question but not covered in the provided passages.

Guidelines:
- Create a single-hop question (answerable by referring to a single piece of information)
- Be specific enough to retrieve relevant information but not overly restrictive
- Maintain important entities and keywords from the original question
- Focus on the most critical missing information
- If multiple pieces of information are missing, prioritize the most fundamental one
- DO NOT use any prior knowledge beyond the provided passages.

You may add some additional explanation or commentary in your response, but make sure to wrap the final retrieval query between <query> and </query> tags.

Here are some examples for your reference:

Example 1:
Question: What's the meaning of the name of the school that does not include the Mahayava scriptures in its canon?
Your Response: The given question is asking about the meaning of the name of a specific school. To answer this, I need to know the name of the school that does not include the Mahayava scriptures in its canon. Therefore, my retrieval query would be: <query>What is the name of the school that does not include the Mahayava scriptures in its canon?</query>

Example 2:
Question: Who founded the company that distributed the film UHF?
Your Response: The given question is asking about the founder of a specific company. To answer this, I need to know the name of the company that distributed the film UHF. Therefore, my retrieval query would be: <query>What is the name of the company that distributed the film UHF?</query>

Example 3:
Question: Who was picked in the NBA draft before the player with the highest point average among active players?
Confirmed Passage 1: Wilt Chamberlain holds the all - time records for total points scored (4,029) and points per game (50.4) in a season; both records were achieved in the 1961 -- 62 season. He also holds the rookie records for points per game when he averaged 37.6 points in the 1959 -- 60 season. Among active players, Kevin Durant has the highest point total (2,593) and the highest scoring average (32.0) in a season; both were achieved in the 2013 -- 14 season.
Your Response: From the given passage, I can see that Kevin Durant has the highest point average among active players. To answer the question, I need to know who was picked in the NBA draft before him. Therefore, my retrieval query would be: <query>Who was picked in the NBA draft before Kevin Durant?</query>

Example 4:
Question: When did the city that WIZE is licensed to broadcast to, become the capital of the state, that contains the county where the Battle of Rich Mountain occurred?
Confirmed Passage 1: The Battle of Rich Mountain took place on July 11, 1861, in Randolph County, Virginia (now West Virginia) as part of the Operations in Western Virginia Campaign during the American Civil War.
Confirmed Passage 2: Owing to its role in the state's history, the county motto is \"Where Illinois Began.\" It contains the historically important village of Kaskaskia, Illinois's first capital.
Confirmed Passage 3: WIZE (1340 AM) — branded WIZE AM 1340 — is a commercial radio station in Springfield, Ohio owned by iHeartMedia, Inc. as part of their Dayton cluster. The station's main format is classic country targeted towards Springfield, and their transmitter - and former studios - are also located in Springfield.
Your Response: From the given passages, I can see that the Battle of Rich Mountain occurred in Randolph County, Virginia (now West Virginia), and that Kaskaskia was Illinois's first capital. To answer the question, I need to know when Springfield became the capital of Illinois. Therefore, my retrieval query would be: <query>When did Springfield become the capital of Illinois?</query>
"""

COT_WO_VERIFIER_SYSTEM_PROMPT = """
You are tasked with generating targeted retrieval queries for multi-hop questions. A multi-hop question requires multiple pieces of information to be answered fully.

Given:
- A multi-hop question
- A set of confirmed, relevant passages (which may be empty)

Your task:
Generate a single retrieval query that targets ONE specific missing or complementary piece of evidence needed to answer the original question that is not covered by the provided passages.

Guidelines:
1. First, perform an internal chain-of-thought (CoT) analysis to determine whether the confirmed passages fully answer the question or if additional information is needed. (Do not include your CoT in the final output.)
2. If the confirmed passages already fully answer the question, your final response should include your internal reasoning and then end with <stop>.
3. If additional information is necessary, your final response should include your internal reasoning and then append a retrieval query enclosed in <query> and </query> tags.
4. Use only the information provided in the confirmed passages; do not rely on any external or prior knowledge.

[Information Sufficiency Checklist]
- Do the confirmed passages contain all critical details (names, dates, events, locations, etc.) required to answer the question?
- If all required details are present, your final response should end with <stop>.

Few-Shot Examples:

Example 1:
Question: What is the capital of France?
Your Response: The question asks for the capital of France, and the confirmed passage clearly states that the capital is Paris. All required details are provided. <stop>

Example 2:
Question: What's the meaning of the name of the school that does not include the Mahayava scriptures in its canon?
Your Response: The question asks for the meaning of the school's name, but the confirmed passage does not provide the actual name of the school. I need the name to determine its meaning. <query>What is the name of the school that does not include the Mahayava scriptures in its canon?</query>

Example 3:
Question: Who was picked in the NBA draft before the player with the highest point average among active players?
Confirmed Passage 1: "Wilt Chamberlain holds the all-time records for total points scored (4,029) and points per game (50.4) in a season; he also holds the rookie record when he averaged 37.6 points in the 1959–60 season. Among active players, Kevin Durant has the highest point average (32.0) in a season, achieved in the 2013–14 season."
Your Response: The question seeks to know who was picked in the NBA draft before the player with the highest point average among active players. The confirmed passage shows that Kevin Durant is that player but does not specify who was drafted before him. <query>Who was picked in the NBA draft before Kevin Durant?</query>

Example 4:
Question: When did the city that WIZE is licensed to broadcast to become the capital of the state that contains the county where the Battle of Rich Mountain occurred?
Confirmed Passage 1: "The Battle of Rich Mountain took place on July 11, 1861, in Randolph County, Virginia (now West Virginia) during the American Civil War."
Confirmed Passage 2: "Kaskaskia, located in that county, was Illinois's first capital."
Confirmed Passage 3: "WIZE (1340 AM) is a radio station based in Springfield, Ohio."
Your Response: The confirmed passages provide historical context and indicate that Kaskaskia was Illinois's first capital, but they do not specify when Springfield became the capital of Illinois. <query>When did Springfield become the capital of Illinois?</query>

Example 5 (2-Hop with Two Confirmed Passages – <stop> Output):
Question: What river flows through Paris, and which iconic monument on its banks was completed in 1889?
Confirmed Passage 1: "The Seine River flows through the heart of Paris."
Confirmed Passage 2: "The Eiffel Tower, an iconic monument along the Seine, was completed in 1889."
Your Response: The question asks for the river flowing through Paris and the iconic monument completed in 1889 on its banks. The confirmed passages provide both details: the Seine River and the Eiffel Tower. All required information is present. <stop>
"""

SELF_ASK_SYSTEM_PROMPT = """
Given the following question, answer it by providing follow up questions and intermediate answers.
For each follow up question, you are given a context which is the top returned Wikipedia snippet for the question.
First give an intermediate answer to the follow up question. Start the line with "Intermediate answer: ".
Then, ask a next follow up question if necessary. Start the line with "Follow up: ".
If no follow up questions are necessary, give a final answer directly. Start the line with "So the final answer is: ".
Output only the process after the given prompt. Do not repeat the given prompt in your response.

#
Question: What is the major railroad museum located in the location where Andre Bloc lived at his time of death?
Follow up: Where did André Bloc live when he died?
Context: André Bloc (Algiers, May 23, 1896 – New Delhi, November 8, 1966) was a French sculptor, magazine editor, and founder of several specialist journals. He founded the "Groupe Espace" in 1949.
Intermediate answer: André Bloc was living in New Delhi when he died.
Follow up: What is the name of the major railroad related museum located in New Delhi?
Context: New Delhi is home to Indira Gandhi Memorial Museum, National Gallery of Modern Art, National Museum of Natural History, National Rail Museum, National Handicrafts and Handlooms Museum, National Philatelic Museum, Nehru Planetarium, Shankar's International Dolls Museum. and Supreme Court of India Museum.
Intermediate answer: The National Rail Museum is located in New Delhi.
So the final answer is: National Rail Museum

#
Question: What is the least popular official language in the country where a spiral viaduct is located in Karin Thomas' birthplace?
Follow up: Where was Karin Thomas born?
Context: Karin Thomas (born 3 October 1961 in Brusio) was a Swiss cross country skier who competed from 1982 to 1988. She finished sixth in the 4 x 5 km relay at the 1984 Winter Olympics in Sarajevo and fourth in that same event at the 1988 Winter Olympics in Calgary.
Intermediate answer: Karin Thomas was born in Brusio.
Follow up: In which country is a spiral viaduct is located in Brusio?
Context: Brusio spiral viaduct: A signature structure of the World Heritage-listed Bernina railway, it is located near Brusio, in the Canton of Graubünden, Switzerland, and was built to limit the railway's gradient at that location within its specified maximum of 7%.
Intermediate answer: The Brusio spiral viaduct is located in Switzerland.
Follow up: What is the least popular official language of Switzerland?
Context: Switzerland has four official languages: principally German (63.5% total population share, with foreign residents, in 2013); French (22.5%) in the west; and Italian (8.1%) in the south. The fourth official language, Romansh (0.5%), is a Romance language spoken locally in the southeastern trilingual canton of Graubünden, and is designated by Article 4 of the Federal Constitution as a national language along with German, French, and Italian, and in Article 70 as an official language if the authorities communicate with persons who speak Romansh. However, federal laws and other official acts do not need to be decreed in Romansh.
Intermediate answer: The least popular official language is Romansh.
So the final answer is: Romansh

#
Question: When did hurricane Sandy his the city where The Dealers' performer was born?
Follow up: Who is the performer of The Dealers?
Context: The Dealers is a 1964 album by jazz musician Mal Waldron released on Status Records, catalogue 8316. The album consists of unreleased takes from two sessions that resulted in two prior albums. "Blue Calypso" and "Falling In Love With Love" are from the April 19, 1957 session that resulted in half of 1957 Waldron's "Mal/2" album; these tracks can currently be found as additional tracks on the CD reissue of that album. "Dealin'" and "Wheelin" are from a September 20, 1957 session, and are alternate takes of tracks originally released on the 1958 "Wheelin' & Dealin'" album (Prestige PRLP 7131); these tracks can currently be found as additional tracks on the CD reissue of that album. All tracks are also available as part of the 2009 John Coltrane's box set "Side Steps".
Intermediate answer: The Dealers is an album by Mal Waldron.
Follow up: Where was Mal Waldron born?
Context: Malcolm Earl "Mal" Waldron (August 16, 1925 – December 2, 2002) was an American jazz pianist, composer, and arranger. Mal Waldron was born in New York City on August 16, 1925, to West Indian immigrants. His father was a mechanical engineer who worked on the Long Island Rail Road. The family moved to Jamaica, Queens when Mal was four years old. Waldron's parents discouraged his initial interest in jazz, but he was able to maintain it by listening to swing on the radio.
Intermediate answer: Mal Waldron was born in New York City.
Follow up: When did hurricane sandy hit New York City?
Context: Effects of Hurricane Sandy in New York: Hurricane Sandy Category 1 hurricane (SSHWS / NWS) Satellite image of Sandy at 4: 15 p.m. EDT on October 29 as it was about to make landfall on the Jersey Shore Formed October 28, 2012 (First rainbands begin to affect New Jersey) Dissipated November 2, 2012 (Dissipated as extratropical cyclone) (Extratropical after October 29) Highest winds 1 - minute sustained: 80 mph (130 km / h) Highest gust Gusts: 100 mph (155 km / h) Lowest pressure 945 mbar (hPa); 27.91 inHg Fatalities 53 total Damage $32 billion (2012 USD) (Estimated damage total) Areas affected New York, especially the New York metropolitan area Part of the 2012 Atlantic hurricane season Part of a series on Hurricane Sandy General Meteorological history Impact Greater Antilles United States Maryland and Washington, D.C. New Jersey New York New England Canada Other wikis Commons: Sandy images Wikinews: Sandy stories
Intermediate answer: Hurricane sandy hit New York City in October 28, 2012.
So the final answer is: October 28, 2012
"""


QUERY_GENERATION_SYSTEM_PROMPT = """
Given an original question and a series of follow-up questions and answers, generate the next logical follow-up question that targets a specific missing piece of information needed to answer the original question.
The question will be used to retrieve additional information from a knowledge base (e.g., Wikipedia).

Process:
- You receive:
  - Main question: <original question>  
  - Zero or more rounds of:
    - Follow up: <previous follow up question>  
    - Document: <top Wikipedia snippet>  
    - Intermediate answer: <answer to the Follow-up question based on the Document>

- Your task:
  1. Ask the next logical follow-up question.
  2. Base it solely on gaps in the existing trace.
  3. Do not repeat information already covered.
  4. Make sure it targets exactly the missing fact you need to answer the original question.
  5. Output only the new question itself; no explanations or extra text.

Output only the process after the given prompt. Do not repeat the given prompt in your response.

Here are some examples:

# First iteration (no trace)
## Input
Main question: What is the major railroad museum located in the location where Andre Bloc lived at his time of death?
## Output
Where did André Bloc live when he died?

# First iteration (no trace)
## Input
Main question: What is the least popular official language in the country where a spiral viaduct is located in Karin Thomas' birthplace?
## Output
Where was Karin Thomas born?

# Subsequent iteration (trace present)
## Input
Main question: What is the major railroad museum located in the location where Andre Bloc lived at his time of death?

Follow up: Where did André Bloc live when he died?
Document: André Bloc (Algiers, May 23, 1896 – New Delhi, November 8, 1966) was a French sculptor, magazine editor, and founder of several specialist journals. He founded the "Groupe Espace" in 1949.
Intermediate answer: André Bloc was living in New Delhi when he died.
## Output
What is the name of the major railroad related museum located in New Delhi?

# Subsequent iteration (trace present)
## Input
Main question: What is the least popular official language in the country where a spiral viaduct is located in Karin Thomas' birthplace?

Follow up: Where was Karin Thomas born?
Document: Karin Thomas (born 3 October 1961 in Brusio) was a Swiss cross country skier who competed from 1982 to 1988. She finished sixth in the 4 x 5 km relay at the 1984 Winter Olympics in Sarajevo and fourth in that same event at the 1988 Winter Olympics in Calgary.
Intermediate answer: Karin Thomas was born in Brusio.
Follow up: In which country is a spiral viaduct is located in Brusio?
Document: Brusio spiral viaduct: A signature structure of the World Heritage-listed Bernina railway, it is located near Brusio, in the Canton of Graubünden, Switzerland, and was built to limit the railway's gradient at that location within its specified maximum of 7%.
Intermediate answer: The Brusio spiral viaduct is located in Switzerland.
## Output
What is the least popular official language of Switzerland?

# Subsequent iteration (trace present)
## Input
Main question: When did hurricane Sandy hit the city where The Dealers' performer was born?

Follow up: Who is the performer of The Dealers?
Document: The Dealers is a 1964 album by jazz musician Mal Waldron released on Status Records, catalogue 8316. The album consists of unreleased takes from two sessions that resulted in two prior albums. "Blue Calypso" and "Falling In Love With Love" are from the April 19, 1957 session that resulted in half of 1957 Waldron's "Mal/2" album; these tracks can currently be found as additional tracks on the CD reissue of that album. "Dealin'" and "Wheelin" are from a September 20, 1957 session, and are alternate takes of tracks originally released on the 1958 "Wheelin' & Dealin'" album (Prestige PRLP 7131); these tracks can currently be found as additional tracks on the CD reissue of that album. All tracks are also available as part of the 2009 John Coltrane's box set "Side Steps".
Intermediate answer: The Dealers is an album by Mal Waldron.
Follow up: Where was Mal Waldron born?
Document: Malcolm Earl "Mal" Waldron (August 16, 1925 – December 2, 2002) was an American jazz pianist, composer, and arranger. Mal Waldron was born in New York City on August 16, 1925, to West Indian immigrants. His father was a mechanical engineer who worked on the Long Island Rail Road. The family moved to Jamaica, Queens when Mal was four years old. Waldron's parents discouraged his initial interest in jazz, but he was able to maintain it by listening to swing on the radio.
Intermediate answer: Mal Waldron was born in New York City.
## Output
When did hurricane sandy hit New York City?
"""


QUERY_GENERATION_USER_PROMPT = "Respond with a simple follow-up question that will help answer the main question, do not explain yourself or output anything else."


def gen_retriever_query_prompt(question, trace):
    chat = [
        {
            "role": "system",
            "content": QUERY_GENERATION_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": "Main question: " + question.strip() + "\n\n" + trace.strip() + "\n\n" + QUERY_GENERATION_USER_PROMPT,
        },
    ]

    return chat
