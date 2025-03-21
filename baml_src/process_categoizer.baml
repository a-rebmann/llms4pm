

class CategorizedProcess {
    process_id string
    process_name string
    unique_activities string[]
    category string
}


function CategorizeProcess(activities: string[], model_name: string, categories: string[], process_id: string) -> CategorizedProcess {
     client Default
     prompt #"
        {{_.role("system")}}
        You are an expert in categorizing processes. 
        Your task is to accept a list containing activity labels such as "create purchase order" and a process name such as "procurement" 
        as input and provide the best category for the given process from the list provided below. If no category fits, return "Other".
        Note: the process name is often not very informative, so you should rather focus on the activities.
        The categories are: 
        {{categories}} 

        {{ctx.output_format}}

        {{_.role("user")}}

        This is the list of activitiy labels
         <labels>{{activities}}</labels>

        This is the process name
         <process_name>{{model_name}}</process_name>

         This is the process id to include in the output for tracking purposes
         <process_id>{{process_id}}</process_id>


        {{_.role("assistant")}}
        JSON:
    "#
}


test process_categorization{
    functions [CategorizeProcess]
    args { 
        activities ["Receive invoice", "Receive materials", "Pay the invoice", "Send Purchase order", "Create Purchase order"]
        model_name "Procurement"
        process_id "1234"
        categories ["Develop and Manage Products and Services",
        "Develop and Manage Human Capital",
        "Develop Vision and Strategy",
        "Market and Sell Products and Services",
        "Deliver Physical Products",
        "Deliver Services",
        "Manage Customer Service",
        "Manage Information Technology",
        "Managee Financial Resources",
        "Acquire, Construct, and Manage Assets",
        "Manage Enterprise Risk, Compliance, Remediation, and Resiliency",
        "Manage External Relationships",
        "Develop and Manage Business Capabilities"]
        }
}

