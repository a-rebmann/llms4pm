class ObjectActionPair{
    object string
    action string
}

class PairsFromLabel{
    original_label string
    object_action_pairs ObjectActionPair[]
}

class ExtractedPairs {
    pairs_from_label PairsFromLabel[]
}


function ExtractObjectActionPairs(labels: string[]) -> ExtractedPairs {
    client Default
    prompt #"
        {{_.role("system")}}
        You are an expert activity label tagger system. 
        Your task is to accept a list containing activity labels such as 'create purchase order' as input and provide a list of pairs, where each pair consists of the main action and the object it is applied on. 
        For 'create purchase order', you would return extract 'create' as the action and 'purchase order' as the object. 
        If actions are not provided as verbs, change them into verbs, e.g., for 'purchase order creation' you would hence return ('create', 'purchase order') as well. 
        Also turn past tense actions into present tense ones, e.g., 'purchase order created' becomes ('create', 'purchase order') too. 
        If multiple actions are applied to the same object, split this into multiple pairs, e.g., 'create and send purchase order' becomes ('create', 'purchase order'), ('send', 'purchase oder')
        If there is additional information, e.g., about who is performing the action or about an IT system that is involved, discard that. 
        If there are any special characters, just replace them with whitespace and turn objects ans actions to all lower case characters. 

        {{ctx.output_format}}

        {{_.role("user")}}

        This is the list of activitiy labels
         <labels>{{labels}}</labels>


        {{_.role("assistant")}}
        JSON:
    "#
}


test label_extraction {
  functions [ExtractObjectActionPairs]
  args {
    labels ["Wellcome Customers",
    "Check if customer is new or existing?",
    "Contacts the patient",
    "Collect medication",
    "Business Model Definition",
    "Step 1C",
    "basic ingredients not accessible",
    "Submit finalised mockup to Liquid Assembly",
    "Write Settlement Recommendation",
    "Notification"]
    
  }
}