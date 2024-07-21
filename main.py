import os
from dotenv import load_dotenv
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
from output_example import output_example_JSON


def pull_content_openAI(technology_tech_course_name: str, lesson_name: str):
    course_page_master_prompt = """
    given the name of a technology/tech course along with the lesson name, mentioned below 
    
    technology/tech course: {technology_tech_course_name}
    lesson name: {lesson_name}
    
    please provide me with a brief explanation on {lesson_name} under {technology_tech_course_name} with a maximum of 10 bullet points and a minimum 1 bullet point 
    
    also, provide code blocks to explain concepts with appropriate comments in the code
    
    provide JSON as output
    below is a OUTPUT example that you should follow:
    {output_example_JSON}
    """

    course_page_master_prompt_template = PromptTemplate(
        input_variables=[
            "technology_tech_course_name",
            "lesson_name",
            "output_example_JSON",
        ],
        template=course_page_master_prompt,
    )

    llm = ChatOpenAI(
        model="gpt-4o", model_kwargs={"response_format": {"type": "json_object"}}
    )

    chain = course_page_master_prompt_template | llm
    res = chain.invoke(
        input={
            "technology_tech_course_name": technology_tech_course_name,
            "lesson_name": lesson_name,
            "output_example_JSON": output_example_JSON,
        }
    )
    return res


if __name__ == "__main__":
    load_dotenv()
    results = pull_content_openAI(
        technology_tech_course_name="typescript for beginners",
        lesson_name="array methods"
    )
    print(results.content)
