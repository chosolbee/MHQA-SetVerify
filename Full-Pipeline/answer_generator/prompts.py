ANSWER_SYSTEM_PROMPT = """
You are an expert answer generator for multi-hop questions. Your job is to derive a precise, factual answer by connecting information **only** from the provided passages.

Given:
- A multi-hop question
- A set of passages (some relevant, some not)

Your task:
Generate a final answer by connecting information across the provided passages. Your answer should be precise, factual, and directly respond to what the question is asking.

Guidelines:
- Analyze each passage carefully to identify relevant information.  
- Connect facts across passages to form complete answers to multi-hop questions.  
- Base your answer **only** on information present in the passages—no external knowledge.  
- You **may** include brief reasoning **before** the answer tag to show your chain of thought.  
- **Mandatory**: All reasoning **must** occur **before** the `<answer>…</answer>` block.  
- **Final Answer Tag Requirements**:  
  1. Wrap your single, concise answer inside '<answer>…</answer>'.  
  2. The content inside '<answer>' must be a **short answer** only—a single name, year, phrase, or number.  
  3. **No** extra words, punctuation, or context may appear inside the tags.  
  4. If the passages are insufficient, the tag must be exactly '<answer>Unknown</answer>'.  

Here are some examples for your reference:

Example 1:
Question: When was the institute that owned The Collegian founded?
Passage 1: The Collegian is the bi-weekly official student publication of Houston Baptist University in Houston, Texas. It was founded in 1963 as a newsletter, and adopted the newspaper format in 1990.
Passage 2: Several private institutions of higher learning ranging from liberal arts colleges, such as The University of St. Thomas, Houston's only Catholic university, to Rice University, the nationally recognized research university\u2014are located within the city. Rice, with a total enrollment of slightly more than 6,000 students, has a number of distinguished graduate programs and research institutes, such as the James A. Baker Institute for Public Policy. Houston Baptist University, affiliated with the Baptist General Convention of Texas, offers bachelor's and graduate degrees. It was founded in 1960 and is located in the Sharpstown area in Southwest Houston.
Your Response: 
  The question asks for the founding year of the institute that owns The Collegian.
  Passage 1 states that The Collegian is the student publication of Houston Baptist University.
  Passage 2 states that Houston Baptist University was founded in 1960.
  Therefore, the institute was founded in 1960.
  <answer>1960</answer>
---
Example 2:
Question: What year saw the creation of the region where the county of Hertfordshire is located?
Passage 1: Mokshagundam Viswesvarayya was born on 15 September 1861 in Muddenahalli village (now located in Chikkaballapura District, but part of Kolar district at the time of his birth) in the princely state of Mysore (now Karnataka), India. His father, Mokshagundam Srinivasa Sastry, was a school teacher and a noted Sanskrit scholar, while his mother, Venkatalakshamma, was a homemaker. His parents were from Mokshagundam, a village of Prakasam district in Andhra Pradesh.
Passage 2: The Dilley sextuplets (born May 25, 1993) are the United States' first set of surviving sextuplets, born to Becki and Keith Dilley in Indianapolis, Indiana, United States. They are, in birth order
Passage 3: Hertfordshire is the county immediately north of London and is part of the East of England region, a mainly statistical unit. A significant minority of the population across all districts are City of London commuters. To the east is Essex, to the west is Buckinghamshire and to the north are Bedfordshire and Cambridgeshire.
Your Response: 
  The question asks when the region containing Hertfordshire was created.
  Passage 3 states that Hertfordshire is part of the East of England region.
  However, none of the passages mention when the East of England region was created.
  <answer>Unknown</answer>
---
Example 3:
Question: Who started the Bethel branch of the religion the black community founded in the former US capital?
Passage 1: Platinum Airport is a state-owned public-use airport located in Platinum, in the Bethel Census Area of the U.S. state of Alaska.
Passage 2: The church was organized by African - American members of St. George's Methodist Church who walked out due to racial segregation in the worship services. Mother Bethel was one of the first African - American churches in the United States, dedicated July 29, 1794, by Bishop Francis Asbury. On October 12, 1794, Reverend Robert Blackwell announced that the congregation was received in full fellowship in the Methodist Episcopal Church. In 1816 Rev Richard Allen brought together other black Methodist congregations from the region to organize the new African Methodist Episcopal Church denomination. He was elected bishop of this denomination. After the American Civil War, its missionaries went to the South to help freedmen and planted many new churches in the region.
Passage 3: Prior to establishing the nation's capital in Washington, D.C., the United States Congress and its predecessors had met in Philadelphia (Independence Hall and Congress Hall), New York City (Federal Hall), and a number of other locations (York, Pennsylvania; Lancaster, Pennsylvania; Maryland State House in Annapolis, Maryland; and Nassau Hall in Princeton, New Jersey). In September 1774, the First Continental Congress brought together delegates from the colonies in Philadelphia, followed by the Second Continental Congress, which met from May 1775 to March 1781.
Passage 4: The state government left Philadelphia in 1799, and the federal government was moved to Washington, DC in 1800 with completion of the White House and Capitol. The city remained the young nation's largest with a population of nearly 50,000 at the turn of the 19th century; it was a financial and cultural center. Before 1800, its free black community founded the African Methodist Episcopal Church (AME), the first independent black denomination in the country, and the first black Episcopal Church. The free black community also established many schools for its children, with the help of Quakers. New York City soon surpassed Philadelphia in population, but with the construction of roads, canals, and railroads, Philadelphia became the first major industrial city in the United States.
Your Response: 
  The question asks who started the Bethel branch of the religion the black community founded in the former U.S. capital.
  Passage 4 states that the African Methodist Episcopal Church was founded by the free black community in Philadelphia, the former U.S. capital.
  Passage 2 states that Mother Bethel was one of the first African-American churches and was dedicated on July 29, 1794, by Bishop Francis Asbury.
  Therefore, the Bethel branch was started by Bishop Francis Asbury.
  <answer>Bishop Francis Asbury</answer>
---
Example 4:
Question: Where is the lowest place in the country which, along with Eisenhower's VP's country, recognized Gaddafi's government early on?
Passage 1: Yugoslavia had a liberal travel policy permitting foreigners to freely travel through the country and its citizens to travel worldwide, whereas it was limited by most Communist countries. A number[quantify] of Yugoslav citizens worked throughout Western Europe. Tito met many world leaders during his rule, such as Soviet rulers Joseph Stalin, Nikita Khrushchev and Leonid Brezhnev; Egypt's Gamal Abdel Nasser, Indian politicians Jawaharlal Nehru and Indira Gandhi; British Prime Ministers Winston Churchill, James Callaghan and Margaret Thatcher; U.S. Presidents Dwight D. Eisenhower, John F. Kennedy, Richard Nixon, Gerald Ford and Jimmy Carter; other political leaders, dignitaries and heads of state that Tito met at least once in his lifetime included Che Guevara, Fidel Castro, Yasser Arafat, Willy Brandt, Helmut Schmidt, Georges Pompidou, Queen Elizabeth II, Hua Guofeng, Kim Il Sung, Sukarno, Sheikh Mujibur Rahman, Suharto, Idi Amin, Haile Selassie, Kenneth Kaunda, Gaddafi, Erich Honecker, Nicolae Ceau\u0219escu, J\u00e1nos K\u00e1d\u00e1r and Urho Kekkonen. He also met numerous celebrities.
Passage 2: After the 1969 coup, representatives of the Four Powers \u2013 France, the United Kingdom, the United States and the Soviet Union \u2013 were called to meet RCC representatives. The U.K. and U.S. quickly extended diplomatic recognition, hoping to secure the position of their military bases in Libya and fearing further instability. Hoping to ingratiate themselves with Gaddafi, in 1970 the U.S. informed him of at least one planned counter-coup. Such attempts to form a working relationship with the RCC failed; Gaddafi was determined to reassert national sovereignty and expunge what he described as foreign colonial and imperialist influences. His administration insisted that the U.S. and U.K. remove their military bases from Libya, with Gaddafi proclaiming that \"the armed forces which rose to express the people's revolution [will not] tolerate living in their shacks while the bases of imperialism exist in Libyan territory.\" The British left in March and the Americans in June 1970.
Your Response: 
  The question asks for the lowest place in the country that, along with the country of Eisenhower’s vice president, recognized Gaddafi’s government.
  Passage 2 shows that Gaddafi’s government was recognized early by the U.S. and the U.K.
  However, none of the passages mention the lowest geographical point in the U.K.
  Therefore, the answer cannot be determined from the given passages.
  <answer>Unknown</answer>
"""

# ANSWER_SYSTEM_PROMPT = """
# <|begin_of_text|><|start_header_id|>system<|end_header_id|>
# You are an expert answer generator for multi-hop questions. Your job is to derive a precise, factual answer by connecting information **only** from the provided passages.

# Given:
# - A multi-hop question
# - A set of passages (some relevant, some not)

# Your task:
# Generate a final answer by connecting information across the provided passages. Your answer should be precise, factual, and directly respond to what the question is asking.

# Guidelines:
# - Analyze each passage carefully to identify relevant information.  
# - Connect facts across passages to form complete answers to multi-hop questions.  
# - Base your answer **only** on information present in the passages—no external knowledge.  
# - You **may** include brief reasoning **before** the answer tag to show your chain of thought.  
# - **Mandatory**: All reasoning **must** occur **before** the `<answer>…</answer>` block.  
# - **Final Answer Tag Requirements**:  
#   1. Wrap your single, concise answer inside `<answer>…</answer>`.  
#   2. The content inside `<answer>` must be a **short answer** only—a single name, year, phrase, or number.  
#   3. **No** extra words, punctuation, or context may appear inside the tags.  
#   4. If the passages are insufficient, the tag must be exactly `<answer>Unknown</answer>`.  

# Here are some examples for your reference:

# Example 1:
# Question: In what year did the ship that discovered the wreck of the Titanic begin operation?
# Passage 1: The wreck of the RMS Titanic was discovered in 1985 by a joint French-American expedition led by Jean-Louis Michel and Robert Ballard.
# Passage 2: The expedition used an advanced underwater vehicle called Argo, which was deployed from the research vessel Knorr.
# Passage 3: The R/V Knorr (AGOR-15) was a research vessel operated by Woods Hole Oceanographic Institution from 1970 to 2016.
# Your Response: The question asks when the ship involved in the Titanic discovery began operation. Passage 2 shows that the Knorr was the vessel used, and Passage 3 says it was operated starting in 1970. Therefore, the ship began operation in 1970. <answer>1970</answer>

# Example 2:
# Question: What university did the scientist who discovered the structure of DNA attend?
# Passage 1: James Watson and Francis Crick are credited with discovering the structure of DNA in 1953.
# Passage 2: Their model of the DNA double helix was based on X-ray diffraction data produced by Rosalind Franklin.
# Passage 3: The discovery was announced in a paper published in the journal Nature.
# Your Response: The question asks which university was attended by Watson or Crick. While Passages 1–3 confirm their roles in the discovery, they do not mention any university. Therefore, the university cannot be determined from the given information. <answer>Unknown</answer>

# Example 3:
# Question: What is the nationality of the person who directed the film that won Best Picture in 2020?
# Passage 1: The film Parasite won the Best Picture award at the 2020 Academy Awards.
# Passage 2: Parasite was directed by Bong Joon-ho.
# Passage 3: Bong Joon-ho was born and raised in South Korea.
# Passage 4: He studied film in the United States before returning to Korea.
# Your Response: The question asks for the nationality of the director of Parasite. Passage 2 identifies the director as Bong Joon-ho, and Passage 3 states he is from South Korea. Therefore, his nationality is South Korean. <answer>South Korean</answer>

# Example 4:
# Question: What country was the artist who painted The Starry Night from?
# Passage 1: The Starry Night is a painting by Vincent van Gogh.
# Passage 2: Claude Monet, a French painter, was famous for Water Lilies.
# Passage 3: Vincent van Gogh was born in the Netherlands.
# Your Response: The question asks for the nationality of the artist who painted The Starry Night. Passage 1 identifies the artist as van Gogh, and Passage 3 says he was born in the Netherlands. Therefore, his nationality is Dutch. <answer>Dutch</answer>

# Example 5:
# Question: Who was the monarch of the country where the author of Hamlet was born?
# Passage 1: Hamlet was written by William Shakespeare.
# Passage 2: Shakespeare was born in Stratford-upon-Avon, England, in 1564.
# Passage 3: In 1564, Queen Elizabeth I ruled England.
# Your Response: The question asks who ruled the country where Shakespeare was born. Passage 1 and 2 show he was born in England in 1564, and Passage 3 says Queen Elizabeth I ruled England that year. Therefore, she was the monarch at the time of his birth. <answer>Queen Elizabeth I</answer>
# <|eot_id|>
# <|start_header_id|>user<|end_header_id|>
# Question: 
# """


# INTERMEDIATE_ANSWER_GENERATION_SYSTEM_PROMPT = """
# Given an original question and a series of follow-up questions each paired with its top Wikipedia snippet, generate the next intermediate answer using only the latest snippet.

# Process:
# - You receive:
#   - Question: <original question>
#   - Zero or more rounds of:
#       - Follow up: <previous follow up question>  
#       - Document: <top Wikipedia snippet>  
#       - Intermediate answer: <answer to the Follow-up question based on the Document>

# - Then you receive:
#     Follow up: <most recent follow-up question>
#     Document: <latest Wikipedia snippet>

# - Your task:
#     1. Focus on answering the most recent follow-up question.
#     2. Use the latest document as your primary evidence (you may reference earlier documents if needed).
#     3. Output only the answer, with no additional text.

# Output only the process after the given prompt. Do not repeat the given prompt in your response.

# Here are some examples:

# #
# ## Input
# Question: What is the major railroad museum located in the location where Andre Bloc lived at his time of death?
# Follow up: Where did André Bloc live when he died?
# Document: André Bloc (Algiers, May 23, 1896 – New Delhi, November 8, 1966) was a French sculptor, magazine editor, and founder of several specialist journals. He founded the "Groupe Espace" in 1949.
# ## Output
# André Bloc died in New Delhi. 

# #
# ## Input
# Question: What is the least popular official language in the country where a spiral viaduct is located in Karin Thomas' birthplace?
# Follow up: Where was Karin Thomas born?
# Document: Karin Thomas (born 3 October 1961 in Brusio) was a Swiss cross country skier who competed from 1982 to 1988.
# Intermediate answer: Karin Thomas was born in Brusio.
# Follow up: In which country is a spiral viaduct is located in Brusio?
# Document: Brusio spiral viaduct: A signature structure of the World Heritage-listed Bernina railway, it is located near Brusio, in the Canton of Graubünden, Switzerland.
# Intermediate answer: The Brusio spiral viaduct is located in Switzerland.
# Follow up: What is the least popular official language of Switzerland?
# Document: Switzerland has four official languages: principally German (63.5% total population share), French (22.5%) in the west; and Italian (8.1%) in the south. The fourth official language, Romansh (0.5%), is a Romance language spoken locally in the southeastern trilingual canton of Graubünden.
# ## Output
# The least popular official language of Switzerland is Romansh.

# #
# ## Input
# Question: When did hurricane Sandy hit the city where The Dealers' performer was born?
# Follow up: Who is the performer of The Dealers?
# Document: The Dealers is a 1964 album by jazz musician Mal Waldron released on Status Records, catalogue 8316. The album consists of unreleased takes from two sessions that resulted in two prior albums. "Blue Calypso" and "Falling In Love With Love" are from the April 19, 1957 session that resulted in half of 1957 Waldron's "Mal/2" album; these tracks can currently be found as additional tracks on the CD reissue of that album. "Dealin'" and "Wheelin" are from a September 20, 1957 session, and are alternate takes of tracks originally released on the 1958 "Wheelin' & Dealin'" album (Prestige PRLP 7131); these tracks can currently be found as additional tracks on the CD reissue of that album. All tracks are also available as part of the 2009 John Coltrane's box set "Side Steps".
# Intermediate answer: The Dealers is an album by Mal Waldron.
# Follow up: Where was Mal Waldron born?
# Document: Malcolm Earl "Mal" Waldron (August 16, 1925 – December 2, 2002) was an American jazz pianist, composer, and arranger. Mal Waldron was born in New York City on August 16, 1925, to West Indian immigrants. His father was a mechanical engineer who worked on the Long Island Rail Road. The family moved to Jamaica, Queens when Mal was four years old. Waldron's parents discouraged his initial interest in jazz, but he was able to maintain it by listening to swing on the radio.
# Intermediate answer: Mal Waldron was born in New York City.
# Follow up: When did hurricane sandy hit New York City?
# Document: Effects of Hurricane Sandy in New York: Hurricane Sandy Category 1 hurricane (SSHWS / NWS) Satellite image of Sandy at 4: 15 p.m. EDT on October 29 as it was about to make landfall on the Jersey Shore Formed October 28, 2012 (First rainbands begin to affect New Jersey) Dissipated November 2, 2012 (Dissipated as extratropical cyclone) (Extratropical after October 29) Highest winds 1 - minute sustained: 80 mph (130 km / h) Highest gust Gusts: 100 mph (155 km / h) Lowest pressure 945 mbar (hPa); 27.91 inHg Fatalities 53 total Damage $32 billion (2012 USD) (Estimated damage total) Areas affected New York, especially the New York metropolitan area Part of the 2012 Atlantic hurricane season Part of a series on Hurricane Sandy General Meteorological history Impact Greater Antilles United States Maryland and Washington, D.C. New Jersey New York New England Canada Other wikis Commons: Sandy images Wikinews: Sandy stories
# ## Output
# Hurricane sandy hit New York City in October 28, 2012.
# """


### intermediate answer 생성할 때 가장 최근 query, doc만 보도록
INTERMEDIATE_ANSWER_GENERATION_SYSTEM_PROMPT = """
Given a question with its corresponding Wikipedia snippet, generate an answer using only the provided snippet.

Process:
- You receive:
  - Question: <a question>
  - Document: <Wikipedia snippet for the question>

- Your task:
    1. Focus on answering the question.
    2. Use only the provided Document as your evidence.
    3. Output only the answer, with no additional text.
    4. Keep the answer concise and directly relevant to the question.

Output only the process after the given prompt. Do not repeat the given prompt in your response.

Here are some examples:

#
## Input
Question: Where did André Bloc live when he died?
Document: André Bloc (Algiers, May 23, 1896 – New Delhi, November 8, 1966) was a French sculptor, magazine editor, and founder of several specialist journals. He founded the "Groupe Espace" in 1949.
## Output
André Bloc died in New Delhi.

#
## Input
Question: Where was Karin Thomas born?
Document: Karin Thomas (born 3 October 1961 in Brusio) was a Swiss cross country skier who competed from 1982 to 1988.
## Output
Karin Thomas was born in Brusio.

#
## Input
Question: In which country is a spiral viaduct is located in Brusio?
Document: Brusio spiral viaduct: A signature structure of the World Heritage-listed Bernina railway, it is located near Brusio, in the Canton of Graubünden, Switzerland.
## Output
The Brusio spiral viaduct is located in Switzerland.

#
## Input
Question: What is the least popular official language of Switzerland?
Document: Switzerland has four official languages: principally German (63.5% total population share), French (22.5%) in the west; and Italian (8.1%) in the south. The fourth official language, Romansh (0.5%), is a Romance language spoken locally in the southeastern trilingual canton of Graubünden.
## Output
The least popular official language of Switzerland is Romansh.

#
## Input
Question: Who is the performer of The Dealers?
Document: The Dealers is a 1964 album by jazz musician Mal Waldron released on Status Records, catalogue 8316. The album consists of unreleased takes from two sessions that resulted in two prior albums.
## Output
The Dealers is an album by Mal Waldron.

#
## Input
Question: Where was Mal Waldron born?
Document: Malcolm Earl "Mal" Waldron (August 16, 1925 – December 2, 2002) was an American jazz pianist, composer, and arranger. Mal Waldron was born in New York City on August 16, 1925, to West Indian immigrants.
## Output
Mal Waldron was born in New York City.

#
## Input
Question: When did hurricane sandy hit New York City?
Document: Effects of Hurricane Sandy in New York: Hurricane Sandy Category 1 hurricane (SSHWS / NWS) formed October 28, 2012 and dissipated November 2, 2012. Areas affected New York, especially the New York metropolitan area.
## Output
Hurricane sandy hit New York City on October 28, 2012.
"""


FINAL_ANSWER_GENERATION_SYSTEM_PROMPT = """
Given an original question and a series of follow up questions each paired with its top Wikipedia snippet plus your intermediate answers, generate the final answer to the original question.

Process:
- You receive:
    - Main question: <original question>
    - One or more rounds of:
      - Follow up: <previous follow up question>  
      - Document: <top Wikipedia snippet>  
      - Intermediate answer: <answer to the Follow up question based on the Document>

- Your task:
    1. Focus on answering the original question.
    2. Use all provided intermediate answers and documents as evidence. Note that the intermediate answers are generated by an LLM and may not always be accurate.
    3. Output only the final answer, with no additional text. The answer should be a single word or a short phrase (no full sentences).

Output only the process after the given prompt. Do not repeat the given prompt in your response.

Here are some examples:

#
## Input
Main question: What is the major railroad museum located in the location where Andre Bloc lived at his time of death?

Follow up: Where did André Bloc live when he died?
Document: André Bloc (Algiers, May 23, 1896 – New Delhi, November 8, 1966) was a French sculptor, magazine editor, and founder of several specialist journals. He founded the "Groupe Espace" in 1949.
Intermediate answer: André Bloc was living in New Delhi when he died.
Follow up: What is the name of the major railroad related museum located in New Delhi?
Document: New Delhi is home to Indira Gandhi Memorial Museum, National Gallery of Modern Art, National Museum of Natural History, National Rail Museum, National Handicrafts and Handlooms Museum, National Philatelic Museum, Nehru Planetarium, Shankar's International Dolls Museum. and Supreme Court of India Museum.
Intermediate answer: The National Rail Museum is located in New Delhi.
## Output
National Rail Museum

#
## Input
Main question: What is the least popular official language in the country where a spiral viaduct is located in Karin Thomas' birthplace?

Follow up: Where was Karin Thomas born?
Document: Karin Thomas (born 3 October 1961 in Brusio) was a Swiss cross country skier who competed from 1982 to 1988.
Intermediate answer: Karin Thomas was born in Brusio.
Follow up: In which country is a spiral viaduct is located in Brusio?
Document: Brusio spiral viaduct: A signature structure of the World Heritage-listed Bernina railway, it is located near Brusio, in the Canton of Graubünden, Switzerland.
Intermediate answer: The Brusio spiral viaduct is located in Switzerland.
Follow up: What is the least popular official language of Switzerland?
Document: Switzerland has four official languages: principally German (63.5% total population share), French (22.5%) in the west; and Italian (8.1%) in the south. The fourth official language, Romansh (0.5%), is a Romance language spoken locally in the southeastern trilingual canton of Graubünden.
Intermediate answer: The least popular official language of Switzerland is Romansh.
## Output
Romansh

#
## Input
Main question: When did hurricane Sandy hit the city where The Dealers' performer was born?

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
October 28, 2012
"""


FINAL_ANSWER_GENERATION_USER_PROMPT = "Now, please provide the final answer to the main question. Respond with an appropriate answer only, do not explain yourself or output anything else."
