import pprint

INSTRUCT_PROMPT_TEMPLATE = """
<|begin_of_text|>
<|start_header_id|>
system
<|end_header_id|>\n\n
{system_message} 
<|eot_id|>
<|start_header_id|>
user
<|end_header_id|>\n\n 
{prompt}
<|eot_id|>
<|start_header_id|>
assistant
<|end_header_id|>\n\n
"""

SYSTEM_MESSAGE = """
You are a world class expert in reading lengthy articles and 
coming up with brief, yet clear and informative summaries.

Here is an example:
---
### Article:

they may be known as masters of disguise , but this little owl could n't conceal his jealousy as he desperately tried to interrupt a brotherly cuddle between his two siblings . as his two siblings settled down to have a rest on their tiny perch , the needy owlet was spotted tactfully creeping along a tree branch before tugging at one of his brother 's wing . the little owl 's sibling rivalry then gets the better of him , as he swoops under his sibling 's wing , prompting an amusing game of tug-of-war . the scene was photographed from just six metres away by 48-year-old dean mason who hid away in a camouflaged shelter , known as a hide , in beaconsfield , buckinghamshire . he said : ` the three siblings are doing exactly what siblings do - playing combined with the odd argument . ' he added : ` every time i capture something unusual , my heart pounds . i feel such an adrenalin rush whilst at the same time hoping i 've made the right choice of camera settings . ' the birds are small owls , a species which typically measures 22cm tall . the bird was introduced to the uk in the 19th century and is most common in central , southern and south eastern england . perched : a little owl gazes in envy as his two siblings huddle together on a small tree branch in beaconsfield , buckinghamshire . ruffling feathers : as he starts moving towards the pair , the nearer sibling cowers under his brother 's wing , turning his back away from the bird . gentle touch : but the little owl 's sibling rivalry gets the better of him , as he start pulling at his sibling 's wing with one small claw . sibling affection : the bird then creeps closer to his sibling , before slowly pecking at one of their ears . the little owls were photographed by sales manager dean mason , 49 . tug of war : the bird continues to gently pull at his sibling 's wing , prompting an amusing game of tug-of-war . mr mason observed the owls from just six metres away by waiting inside a camouflaged shelter known as a hide . getting in a flap : the determined bird then sneaks under its sibling 's wing . the small owl typically measures 22cm tall . owl get you in the end : the little owl finally decides to dive in head first - much to the visible shock of his siblings .

### Summary:

a little owl was captured trying to join his siblings ' huddle by gently tugging on their wings in beaconsfield , bucks . photos taken by sales manager dean mason , 48 , from bournemouth , from a camouflaged shelter six metres away .
---

Only return a succinct summary.  Do not provide any explanations.
"""

USER_MESSAGE_TEMPLATE = """
Summarize this following article:

### Article: {article}

### Summary:
"""


def batch_transform(batch):
    pprint.pprint(batch)
    return batch
