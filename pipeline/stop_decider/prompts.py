# STOP_DECISION_SYSTEM_PROMPT = """
# Given an original question and a series of follow-up questions each paired with its top Wikipedia snippet plus intermediate answers, determine whether there is sufficient information to provide a complete and accurate final answer to the original question.

# Process:
# - You receive:
#   - Question: <original question>
#   - One or more rounds of:
#       - Follow up: <follow-up question>  
#       - Document: <top Wikipedia snippet>  
#       - Intermediate answer: <answer to the follow-up question based on the Document>

# - Your task:
#     1. Thoroughly analyze whether all components of the original question are answered.
#     2. Verify that the logical chain connecting sub-questions to the original question is complete.
#     3. Assess whether the current information provides a clear path to the final answer.
#     4. Output exactly one line: either "<STOP>" or "<CONTINUE>" with no additional text.

# Decision criteria:
# - <STOP>: All components of the original question can be answered with current information such that a complete and final answer can be derived from it.
# - <CONTINUE>: Missing any required information needed to answer the original question.

# Output only the decision after the given prompt. Do not repeat the given prompt in your response.

# Example:

# #
# Question: What is the major railroad museum located in the location where Andre Bloc lived at his time of death?
# Follow up: Where did André Bloc live when he died?
# Document: André Bloc (Algiers, May 23, 1896 – New Delhi, November 8, 1966) was a French sculptor, magazine editor, and founder of several specialist journals. He founded the "Groupe Espace" in 1949.
# Intermediate answer: André Bloc was living in New Delhi when he died.
# <CONTINUE>

# #
# Question: What is the major railroad museum located in the location where Andre Bloc lived at his time of death?
# Follow up: Where did André Bloc live when he died?
# Document: André Bloc (Algiers, May 23, 1896 – New Delhi, November 8, 1966) was a French sculptor, magazine editor, and founder of several specialist journals. He founded the "Groupe Espace" in 1949.
# Intermediate answer: André Bloc was living in New Delhi when he died.
# Follow up: What is the name of the major railroad related museum located in New Delhi?
# Document: New Delhi is home to Indira Gandhi Memorial Museum, National Gallery of Modern Art, National Museum of Natural History, National Rail Museum, National Handicrafts and Handlooms Museum, National Philatelic Museum, Nehru Planetarium, Shankar's International Dolls Museum. and Supreme Court of India Museum.
# Intermediate answer: The National Rail Museum is located in New Delhi.
# <STOP>

# #
# Question: What is the least popular official language in the country where a spiral viaduct is located in Karin Thomas' birthplace?
# Follow up: Where was Karin Thomas born?
# Document: Karin Thomas (born 3 October 1961 in Brusio) was a Swiss cross country skier who competed from 1982 to 1988. She finished sixth in the 4 x 5 km relay at the 1984 Winter Olympics in Sarajevo and fourth in that same event at the 1988 Winter Olympics in Calgary.
# Intermediate answer: Karin Thomas was born in Brusio.
# Follow up: In which country is a spiral viaduct is located in Brusio?
# Document: Brusio spiral viaduct: A signature structure of the World Heritage-listed Bernina railway, it is located near Brusio, in the Canton of Graubünden, Switzerland, and was built to limit the railway's gradient at that location within its specified maximum of 7%.
# Intermediate answer: The Brusio spiral viaduct is located in Switzerland.
# <CONTINUE>

# #
# Question: What is the least popular official language in the country where a spiral viaduct is located in Karin Thomas' birthplace?
# Follow up: Where was Karin Thomas born?
# Document: Karin Thomas (born 3 October 1961 in Brusio) was a Swiss cross country skier who competed from 1982 to 1988. She finished sixth in the 4 x 5 km relay at the 1984 Winter Olympics in Sarajevo and fourth in that same event at the 1988 Winter Olympics in Calgary.
# Intermediate answer: Karin Thomas was born in Brusio.
# Follow up: In which country is a spiral viaduct is located in Brusio?
# Document: Brusio spiral viaduct: A signature structure of the World Heritage-listed Bernina railway, it is located near Brusio, in the Canton of Graubünden, Switzerland, and was built to limit the railway's gradient at that location within its specified maximum of 7%.
# Intermediate answer: The Brusio spiral viaduct is located in Switzerland.
# Follow up: What is the least popular official language of Switzerland?
# Document: Switzerland has four official languages: principally German (63.5% total population share, with foreign residents, in 2013); French (22.5%) in the west; and Italian (8.1%) in the south. The fourth official language, Romansh (0.5%), is a Romance language spoken locally in the southeastern trilingual canton of Graubünden, and is designated by Article 4 of the Federal Constitution as a national language along with German, French, and Italian, and in Article 70 as an official language if the authorities communicate with persons who speak Romansh. However, federal laws and other official acts do not need to be decreed in Romansh.
# Intermediate answer: The least popular official language is Romansh.
# <STOP>

# #
# Question: When did hurricane Sandy his the city where The Dealers' performer was born?
# Follow up: Who is the performer of The Dealers?
# Document: The Dealers is a 1964 album by jazz musician Mal Waldron released on Status Records, catalogue 8316. The album consists of unreleased takes from two sessions that resulted in two prior albums. "Blue Calypso" and "Falling In Love With Love" are from the April 19, 1957 session that resulted in half of 1957 Waldron's "Mal/2" album; these tracks can currently be found as additional tracks on the CD reissue of that album. "Dealin'" and "Wheelin" are from a September 20, 1957 session, and are alternate takes of tracks originally released on the 1958 "Wheelin' & Dealin'" album (Prestige PRLP 7131); these tracks can currently be found as additional tracks on the CD reissue of that album. All tracks are also available as part of the 2009 John Coltrane's box set "Side Steps".
# Intermediate answer: The Dealers is an album by Mal Waldron.
# <CONTINUE>

# #
# Question: When did hurricane Sandy his the city where The Dealers' performer was born?
# Follow up: Who is the performer of The Dealers?
# Document: The Dealers is a 1964 album by jazz musician Mal Waldron released on Status Records, catalogue 8316. The album consists of unreleased takes from two sessions that resulted in two prior albums. "Blue Calypso" and "Falling In Love With Love" are from the April 19, 1957 session that resulted in half of 1957 Waldron's "Mal/2" album; these tracks can currently be found as additional tracks on the CD reissue of that album. "Dealin'" and "Wheelin" are from a September 20, 1957 session, and are alternate takes of tracks originally released on the 1958 "Wheelin' & Dealin'" album (Prestige PRLP 7131); these tracks can currently be found as additional tracks on the CD reissue of that album. All tracks are also available as part of the 2009 John Coltrane's box set "Side Steps".
# Intermediate answer: The Dealers is an album by Mal Waldron.
# Follow up: Where was Mal Waldron born?
# Document: Malcolm Earl "Mal" Waldron (August 16, 1925 – December 2, 2002) was an American jazz pianist, composer, and arranger. Mal Waldron was born in New York City on August 16, 1925, to West Indian immigrants. His father was a mechanical engineer who worked on the Long Island Rail Road. The family moved to Jamaica, Queens when Mal was four years old. Waldron's parents discouraged his initial interest in jazz, but he was able to maintain it by listening to swing on the radio.
# Intermediate answer: Mal Waldron was born in New York City.
# Follow up: When did hurricane sandy hit New York City?
# Document: Effects of Hurricane Sandy in New York: Hurricane Sandy Category 1 hurricane (SSHWS / NWS) Satellite image of Sandy at 4: 15 p.m. EDT on October 29 as it was about to make landfall on the Jersey Shore Formed October 28, 2012 (First rainbands begin to affect New Jersey) Dissipated November 2, 2012 (Dissipated as extratropical cyclone) (Extratropical after October 29) Highest winds 1 - minute sustained: 80 mph (130 km / h) Highest gust Gusts: 100 mph (155 km / h) Lowest pressure 945 mbar (hPa); 27.91 inHg Fatalities 53 total Damage $32 billion (2012 USD) (Estimated damage total) Areas affected New York, especially the New York metropolitan area Part of the 2012 Atlantic hurricane season Part of a series on Hurricane Sandy General Meteorological history Impact Greater Antilles United States Maryland and Washington, D.C. New Jersey New York New England Canada Other wikis Commons: Sandy images Wikinews: Sandy stories
# Intermediate answer: Hurricane sandy hit New York City in October 28, 2012.
# <STOP>
# """

STOP_DECISION_SYSTEM_PROMPT = """
Given an original question and a series of follow-up questions each paired with its top Wikipedia snippet plus intermediate answers, analyze step by step whether there is sufficient information to provide a complete and accurate final answer to the original question.

Process:
- You receive:
    - Main question: <original question>
    - One or more rounds of:
      - Follow up: <follow-up question>  
      - Document: <top Wikipedia snippet>  
      - Intermediate answer: <answer to the follow-up question based on the Document>

- Your task:
    1. Think step by step about what information is needed to answer the original question
    2. Analyze what information has been gathered so far
    3. Determine if any critical information is still missing
    4. Make a reasoned decision about whether to stop or continue

Format your response as:
Analysis: [Step-by-step reasoning about information completeness]
Decision: <STOP> or <CONTINUE>

Decision criteria:
- <STOP>: All components of the original question can be answered with current information such that a complete and final answer can be derived from it.
- <CONTINUE>: Missing any required information needed to answer the original question.

Output only the analysis and decision after the given prompt. Do not repeat the given prompt in your response.

Example:

#
## Input
Main Question: What is the major railroad museum located in the location where Andre Bloc lived at his time of death?

Follow up: Where did André Bloc live when he died?
Document: André Bloc (Algiers, May 23, 1896 – New Delhi, November 8, 1966) was a French sculptor, magazine editor, and founder of several specialist journals. He founded the "Groupe Espace" in 1949.
Intermediate answer: André Bloc died in New Delhi.
## Output
Analysis: To answer the original question, I need two pieces of information: (1) Where André Bloc lived when he died, and (2) What is the major railroad museum in that location. I have found that André Bloc died in New Delhi, which answers the first part. However, I still need to find information about the major railroad museum located in New Delhi.
Decision: <CONTINUE>

#
## Input
Main Question: What is the major railroad museum located in the location where Andre Bloc lived at his time of death?

Follow up: Where did André Bloc live when he died?
Document: André Bloc (Algiers, May 23, 1896 – New Delhi, November 8, 1966) was a French sculptor, magazine editor, and founder of several specialist journals. He founded the "Groupe Espace" in 1949.
Intermediate answer: André Bloc died in New Delhi.
Follow up: What is the name of the major railroad related museum located in New Delhi?
Document: New Delhi is home to Indira Gandhi Memorial Museum, National Gallery of Modern Art, National Museum of Natural History, National Rail Museum, National Handicrafts and Handlooms Museum, National Philatelic Museum, Nehru Planetarium, Shankar's International Dolls Museum. and Supreme Court of India Museum.
Intermediate answer: The National Rail Museum is located in New Delhi.
## Output
Analysis: To answer the original question, I needed: (1) Where André Bloc lived when he died - I found this is New Delhi, and (2) What is the major railroad museum in that location - I found this is the National Rail Museum. I now have all the information needed to provide a complete answer to the original question.
Decision: <STOP>

#
## Input
Main Question: What is the least popular official language in the country where a spiral viaduct is located in Karin Thomas' birthplace?

Follow up: Where was Karin Thomas born?
Document: Karin Thomas (born 3 October 1961 in Brusio) was a Swiss cross country skier who competed from 1982 to 1988. She finished sixth in the 4 x 5 km relay at the 1984 Winter Olympics in Sarajevo and fourth in that same event at the 1988 Winter Olympics in Calgary.
Intermediate answer: Karin Thomas was born in Brusio.
Follow up: In which country is a spiral viaduct is located in Brusio?
Document: Brusio spiral viaduct: A signature structure of the World Heritage-listed Bernina railway, it is located near Brusio, in the Canton of Graubünden, Switzerland, and was built to limit the railway's gradient at that location within its specified maximum of 7%.
Intermediate answer: The Brusio spiral viaduct is located in Switzerland.
## Output
Analysis: To answer the original question, I need: (1) Karin Thomas' birthplace - found to be Brusio, (2) The country where a spiral viaduct is located in Brusio - found to be Switzerland, and (3) The least popular official language in Switzerland. I have the first two pieces of information but still need to find out what the least popular official language of Switzerland is.
Decision: <CONTINUE>

#
## Input
Main Question: What is the least popular official language in the country where a spiral viaduct is located in Karin Thomas' birthplace?

Follow up: Where was Karin Thomas born?
Document: Karin Thomas (born 3 October 1961 in Brusio) was a Swiss cross country skier who competed from 1982 to 1988. She finished sixth in the 4 x 5 km relay at the 1984 Winter Olympics in Sarajevo and fourth in that same event at the 1988 Winter Olympics in Calgary.
Intermediate answer: Karin Thomas was born in Brusio.
Follow up: In which country is a spiral viaduct is located in Brusio?
Document: Brusio spiral viaduct: A signature structure of the World Heritage-listed Bernina railway, it is located near Brusio, in the Canton of Graubünden, Switzerland, and was built to limit the railway's gradient at that location within its specified maximum of 7%.
Intermediate answer: The Brusio spiral viaduct is located in Switzerland.
Follow up: What is the least popular official language of Switzerland?
Document: Switzerland has four official languages: principally German (63.5% total population share, with foreign residents, in 2013); French (22.5%) in the west; and Italian (8.1%) in the south. The fourth official language, Romansh (0.5%), is a Romance language spoken locally in the southeastern trilingual canton of Graubünden, and is designated by Article 4 of the Federal Constitution as a national language along with German, French, and Italian, and in Article 70 as an official language if the authorities communicate with persons who speak Romansh. However, federal laws and other official acts do not need to be decreed in Romansh.
Intermediate answer: The least popular official language is Romansh.
## Output
Analysis: To answer the original question, I needed: (1) Karin Thomas' birthplace - Brusio, (2) The country where a spiral viaduct is located in Brusio - Switzerland, and (3) The least popular official language in Switzerland - Romansh (0.5% of population). I now have all three pieces of information needed to provide a complete answer.
Decision: <STOP>

#
## Input
Main Question: When did hurricane Sandy hit the city where The Dealers' performer was born?

Follow up: Who is the performer of The Dealers?
Document: The Dealers is a 1964 album by jazz musician Mal Waldron released on Status Records, catalogue 8316. The album consists of unreleased takes from two sessions that resulted in two prior albums. "Blue Calypso" and "Falling In Love With Love" are from the April 19, 1957 session that resulted in half of 1957 Waldron's "Mal/2" album; these tracks can currently be found as additional tracks on the CD reissue of that album. "Dealin'" and "Wheelin" are from a September 20, 1957 session, and are alternate takes of tracks originally released on the 1958 "Wheelin' & Dealin'" album (Prestige PRLP 7131); these tracks can currently be found as additional tracks on the CD reissue of that album. All tracks are also available as part of the 2009 John Coltrane's box set "Side Steps".
Intermediate answer: The Dealers is an album by Mal Waldron.
## Output
Analysis: To answer the original question, I need: (1) Who is the performer of The Dealers - I found this is Mal Waldron, (2) Where was Mal Waldron born, and (3) When did hurricane Sandy hit that city. I have the first piece of information but still need to find Mal Waldron's birthplace and then when hurricane Sandy hit that city.
Decision: <CONTINUE>

#
## Input
Main Question: When did hurricane Sandy hit the city where The Dealers' performer was born?

Follow up: Who is the performer of The Dealers?
Document: The Dealers is a 1964 album by jazz musician Mal Waldron released on Status Records, catalogue 8316. The album consists of unreleased takes from two sessions that resulted in two prior albums. "Blue Calypso" and "Falling In Love With Love" are from the April 19, 1957 session that resulted in half of 1957 Waldron's "Mal/2" album; these tracks can currently be found as additional tracks on the CD reissue of that album. "Dealin'" and "Wheelin" are from a September 20, 1957 session, and are alternate takes of tracks originally released on the 1958 "Wheelin' & Dealin'" album (Prestige PRLP 7131); these tracks can currently be found as additional tracks on the CD reissue of that album. All tracks are also available as part of the 2009 John Coltrane's box set "Side Steps".
Intermediate answer: The Dealers is an album by Mal Waldron.
Follow up: Where was Mal Waldron born?
Document: Malcolm Earl "Mal" Waldron (August 16, 1925 – December 2, 2002) was an American jazz pianist, composer, and arranger. Mal Waldron was born in New York City on August 16, 1925, to West Indian immigrants. His father was a mechanical engineer who worked on the Long Island Rail Road. The family moved to Jamaica, Queens when Mal was four years old. Waldron's parents discouraged his initial interest in jazz, but he was able to maintain it by listening to swing on the radio.
Intermediate answer: Mal Waldron was born in New York City.
## Output
Analysis: To answer the original question, I need: (1) Who is the performer of The Dealers - Mal Waldron, (2) Where was Mal Waldron born - New York City, and (3) When did hurricane Sandy hit New York City. I have the first two pieces of information but still need to find when hurricane Sandy hit New York City.
Decision: <CONTINUE>

#
## Input
Main Question: When did hurricane Sandy hit the city where The Dealers' performer was born?

Follow up: Who is the performer of The Dealers?
Document: The Dealers is a 1964 album by jazz musician Mal Waldron released on Status Records, catalogue 8316. The album consists of unreleased takes from two sessions that resulted in two prior albums. "Blue Calypso" and "Falling In Love With Love" are from the April 19, 1957 session that resulted in half of 1957 Waldron's "Mal/2" album; these tracks can currently be found as additional tracks on the CD reissue of that album. "Dealin'" and "Wheelin" are from a September 20, 1957 session, and are alternate takes of tracks originally released on the 1958 "Wheelin' & Dealin'" album (Prestige PRLP 7131); these tracks can currently be found as additional tracks on the CD reissue of that album. All tracks are also available as part of the 2009 John Coltrane's box set "Side Steps".
Intermediate answer: The Dealers is an album by Mal Waldron.
Follow up: Where was Mal Waldron born?
Document: Malcolm Earl "Mal" Waldron (August 16, 1925 – December 2, 2002) was an American jazz pianist, composer, and arranger. Mal Waldron was born in New York City on August 16, 1925, to West Indian immigrants. His father was a mechanical engineer who worked on the Long Island Rail Road. The family moved to Jamaica, Queens when Mal was four years old. Waldron's parents discouraged his initial interest in jazz, but he was able to maintain it by listening to swing on the radio.
Intermediate answer: Mal Waldron was born in New York City.
Follow up: When did hurricane sandy hit New York City?
Document: Effects of Hurricane Sandy in New York: Hurricane Sandy Category 1 hurricane (SSHWS / NWS) Satellite image of Sandy at 4: 15 p.m. EDT on October 29 as it was about to make landfall on the Jersey Shore Formed October 28, 2012 (First rainbands begin to affect New Jersey) Dissipated November 2, 2012 (Dissipated as extratropical cyclone) (Extratropical after October 29) Highest winds 1 - minute sustained: 80 mph (130 km / h) Highest gust Gusts: 100 mph (155 km / h) Lowest pressure 945 mbar (hPa); 27.91 inHg Fatalities 53 total Damage $32 billion (2012 USD) (Estimated damage total) Areas affected New York, especially the New York metropolitan area Part of the 2012 Atlantic hurricane season Part of a series on Hurricane Sandy General Meteorological history Impact Greater Antilles United States Maryland and Washington, D.C. New Jersey New York New England Canada Other wikis Commons: Sandy images Wikinews: Sandy stories
Intermediate answer: Hurricane sandy hit New York City in October 28, 2012.
## Output
Analysis: To answer the original question, I needed: (1) Who is the performer of The Dealers - Mal Waldron, (2) Where was Mal Waldron born - New York City, and (3) When did hurricane Sandy hit New York City - October 28, 2012. I now have all three pieces of information needed to provide a complete answer.
Decision: <STOP>
"""


STOP_DECISION_DOCS_SYSTEM_PROMPT = """
Given an original question and a series of documents, analyze step by step whether there is sufficient information to provide a complete and accurate final answer to the original question.

Process:
- You receive:
    - Main question: <original question>
    - One or more documents in the format:
      - Document: <Wikipedia snippet>

- Your task:
    1. Think step by step about what information is needed to answer the original question
    2. Analyze what information has been gathered so far
    3. Determine if any critical information is still missing
    4. Make a reasoned decision about whether to stop or continue

Format your response as:
Analysis: [Step-by-step reasoning about information completeness]
Decision: <STOP> or <CONTINUE>

Decision criteria:
- <STOP>: All components of the original question can be answered with current information such that a complete and final answer can be derived from it.
- <CONTINUE>: Missing any required information needed to answer the original question.

Output only the analysis and decision after the given prompt. Do not repeat the given prompt in your response.

Example:

#
## Input
Main Question: What is the major railroad museum located in the location where Andre Bloc lived at his time of death?

Document: André Bloc (Algiers, May 23, 1896 – New Delhi, November 8, 1966) was a French sculptor, magazine editor, and founder of several specialist journals. He founded the "Groupe Espace" in 1949.
## Output
Analysis: To answer the original question, I need two pieces of information: (1) Where André Bloc lived when he died, and (2) What is the major railroad museum in that location. I have found that André Bloc died in New Delhi, which answers the first part. However, I still need to find information about the major railroad museum located in New Delhi.
Decision: <CONTINUE>

#
## Input
Main Question: What is the major railroad museum located in the location where Andre Bloc lived at his time of death?

Document: André Bloc (Algiers, May 23, 1896 – New Delhi, November 8, 1966) was a French sculptor, magazine editor, and founder of several specialist journals. He founded the "Groupe Espace" in 1949.
Document: New Delhi is home to Indira Gandhi Memorial Museum, National Gallery of Modern Art, National Museum of Natural History, National Rail Museum, National Handicrafts and Handlooms Museum, National Philatelic Museum, Nehru Planetarium, Shankar's International Dolls Museum. and Supreme Court of India Museum.
## Output
Analysis: To answer the original question, I needed: (1) Where André Bloc lived when he died - I found this is New Delhi, and (2) What is the major railroad museum in that location - I found this is the National Rail Museum. I now have all the information needed to provide a complete answer to the original question.
Decision: <STOP>

#
## Input
Main Question: What is the least popular official language in the country where a spiral viaduct is located in Karin Thomas' birthplace?

Document: Karin Thomas (born 3 October 1961 in Brusio) was a Swiss cross country skier who competed from 1982 to 1988. She finished sixth in the 4 x 5 km relay at the 1984 Winter Olympics in Sarajevo and fourth in that same event at the 1988 Winter Olympics in Calgary.
Document: Brusio spiral viaduct: A signature structure of the World Heritage-listed Bernina railway, it is located near Brusio, in the Canton of Graubünden, Switzerland, and was built to limit the railway's gradient at that location within its specified maximum of 7%.
## Output
Analysis: To answer the original question, I need: (1) Karin Thomas' birthplace - found to be Brusio, (2) The country where a spiral viaduct is located in Brusio - found to be Switzerland, and (3) The least popular official language in Switzerland. I have the first two pieces of information but still need to find out what the least popular official language of Switzerland is.
Decision: <CONTINUE>

#
## Input
Main Question: What is the least popular official language in the country where a spiral viaduct is located in Karin Thomas' birthplace?

Document: Karin Thomas (born 3 October 1961 in Brusio) was a Swiss cross country skier who competed from 1982 to 1988. She finished sixth in the 4 x 5 km relay at the 1984 Winter Olympics in Sarajevo and fourth in that same event at the 1988 Winter Olympics in Calgary.
Document: Brusio spiral viaduct: A signature structure of the World Heritage-listed Bernina railway, it is located near Brusio, in the Canton of Graubünden, Switzerland, and was built to limit the railway's gradient at that location within its specified maximum of 7%.
Document: Switzerland has four official languages: principally German (63.5% total population share, with foreign residents, in 2013); French (22.5%) in the west; and Italian (8.1%) in the south. The fourth official language, Romansh (0.5%), is a Romance language spoken locally in the southeastern trilingual canton of Graubünden, and is designated by Article 4 of the Federal Constitution as a national language along with German, French, and Italian, and in Article 70 as an official language if the authorities communicate with persons who speak Romansh. However, federal laws and other official acts do not need to be decreed in Romansh.
## Output
Analysis: To answer the original question, I needed: (1) Karin Thomas' birthplace - Brusio, (2) The country where a spiral viaduct is located in Brusio - Switzerland, and (3) The least popular official language in Switzerland - Romansh (0.5% of population). I now have all three pieces of information needed to provide a complete answer.
Decision: <STOP>

#
## Input
Main Question: When did hurricane Sandy hit the city where The Dealers' performer was born?

Document: The Dealers is a 1964 album by jazz musician Mal Waldron released on Status Records, catalogue 8316. The album consists of unreleased takes from two sessions that resulted in two prior albums. "Blue Calypso" and "Falling In Love With Love" are from the April 19, 1957 session that resulted in half of 1957 Waldron's "Mal/2" album; these tracks can currently be found as additional tracks on the CD reissue of that album. "Dealin'" and "Wheelin" are from a September 20, 1957 session, and are alternate takes of tracks originally released on the 1958 "Wheelin' & Dealin'" album (Prestige PRLP 7131); these tracks can currently be found as additional tracks on the CD reissue of that album. All tracks are also available as part of the 2009 John Coltrane's box set "Side Steps".
## Output
Analysis: To answer the original question, I need: (1) Who is the performer of The Dealers - I found this is Mal Waldron, (2) Where was Mal Waldron born, and (3) When did hurricane Sandy hit that city. I have the first piece of information but still need to find Mal Waldron's birthplace and then when hurricane Sandy hit that city.
Decision: <CONTINUE>

#
## Input
Main Question: When did hurricane Sandy hit the city where The Dealers' performer was born?

Document: The Dealers is a 1964 album by jazz musician Mal Waldron released on Status Records, catalogue 8316. The album consists of unreleased takes from two sessions that resulted in two prior albums. "Blue Calypso" and "Falling In Love With Love" are from the April 19, 1957 session that resulted in half of 1957 Waldron's "Mal/2" album; these tracks can currently be found as additional tracks on the CD reissue of that album. "Dealin'" and "Wheelin" are from a September 20, 1957 session, and are alternate takes of tracks originally released on the 1958 "Wheelin' & Dealin'" album (Prestige PRLP 7131); these tracks can currently be found as additional tracks on the CD reissue of that album. All tracks are also available as part of the 2009 John Coltrane's box set "Side Steps".
Document: Malcolm Earl "Mal" Waldron (August 16, 1925 – December 2, 2002) was an American jazz pianist, composer, and arranger. Mal Waldron was born in New York City on August 16, 1925, to West Indian immigrants. His father was a mechanical engineer who worked on the Long Island Rail Road. The family moved to Jamaica, Queens when Mal was four years old. Waldron's parents discouraged his initial interest in jazz, but he was able to maintain it by listening to swing on the radio.
## Output
Analysis: To answer the original question, I need: (1) Who is the performer of The Dealers - Mal Waldron, (2) Where was Mal Waldron born - New York City, and (3) When did hurricane Sandy hit New York City. I have the first two pieces of information but still need to find when hurricane Sandy hit New York City.
Decision: <CONTINUE>

#
## Input
Main Question: When did hurricane Sandy hit the city where The Dealers' performer was born?

Document: The Dealers is a 1964 album by jazz musician Mal Waldron released on Status Records, catalogue 8316. The album consists of unreleased takes from two sessions that resulted in two prior albums. "Blue Calypso" and "Falling In Love With Love" are from the April 19, 1957 session that resulted in half of 1957 Waldron's "Mal/2" album; these tracks can currently be found as additional tracks on the CD reissue of that album. "Dealin'" and "Wheelin" are from a September 20, 1957 session, and are alternate takes of tracks originally released on the 1958 "Wheelin' & Dealin'" album (Prestige PRLP 7131); these tracks can currently be found as additional tracks on the CD reissue of that album. All tracks are also available as part of the 2009 John Coltrane's box set "Side Steps".
Document: Malcolm Earl "Mal" Waldron (August 16, 1925 – December 2, 2002) was an American jazz pianist, composer, and arranger. Mal Waldron was born in New York City on August 16, 1925, to West Indian immigrants. His father was a mechanical engineer who worked on the Long Island Rail Road. The family moved to Jamaica, Queens when Mal was four years old. Waldron's parents discouraged his initial interest in jazz, but he was able to maintain it by listening to swing on the radio.
Document: Effects of Hurricane Sandy in New York: Hurricane Sandy Category 1 hurricane (SSHWS / NWS) Satellite image of Sandy at 4: 15 p.m. EDT on October 29 as it was about to make landfall on the Jersey Shore Formed October 28, 2012 (First rainbands begin to affect New Jersey) Dissipated November 2, 2012 (Dissipated as extratropical cyclone) (Extratropical after October 29) Highest winds 1 - minute sustained: 80 mph (130 km / h) Highest gust Gusts: 100 mph (155 km / h) Lowest pressure 945 mbar (hPa); 27.91 inHg Fatalities 53 total Damage $32 billion (2012 USD) (Estimated damage total) Areas affected New York, especially the New York metropolitan area Part of the 2012 Atlantic hurricane season Part of a series on Hurricane Sandy General Meteorological history Impact Greater Antilles United States Maryland and Washington, D.C. New Jersey New York New England Canada Other wikis Commons: Sandy images Wikinews: Sandy stories
## Output
Analysis: To answer the original question, I needed: (1) Who is the performer of The Dealers - Mal Waldron, (2) Where was Mal Waldron born - New York City, and (3) When did hurricane Sandy hit New York City - October 28, 2012. I now have all three pieces of information needed to provide a complete answer.
Decision: <STOP>
"""


STOP_DECISION_USER_PROMPT = "Based on the information provided, give a short analysis of whether the original question can be answered. Then, decide whether to stop or continue gathering information"


def gen_stop_decision_prompt(question, trace):
    chat = [
        {
            "role": "system",
            "content": STOP_DECISION_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": "Main question: " + question.strip() + "\n\n" + trace.strip() + "\n\n" + STOP_DECISION_USER_PROMPT,
        },
    ]

    return chat


def gen_stop_decision_docs_only_prompt(question: str, trace: str) -> str:
    chat = [
        {
            "role": "system",
            "content": STOP_DECISION_DOCS_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": "Main question: " + question.strip() + "\n\n" + trace.strip() + "\n\n" + STOP_DECISION_USER_PROMPT,
        },
    ]

    return chat
