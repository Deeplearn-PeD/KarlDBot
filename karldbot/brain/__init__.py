 from base_agent.llminterface import LangModel, StructuredLangModel

class Koder:
    def __init__(self, language_model):
        """
        Initialize the Koder with a language model and a prompt manager.

        :param language_model: An instance of a language model (e.g., LangModel, StructuredLangModel).
        :param prompt_manager: An instance of a prompt manager.
        """
        self.language_model = language_model
        self.prompt_manager = PromptManager(LangModel())

    def write_code(self, task_description):
        """
        Write code to accomplish the given task.

        :param task_description: A description of the coding task.
        :return: The generated code.
        """
        prompt = self.prompt_manager.generate_code_writing_prompt(task_description)
        code = self.language_model.ask(prompt)
        return code

    def debug_code(self, code_snippet):
        """
        Debug the given code snippet.

        :param code_snippet: The code snippet to be debugged.
        :return: The debugged code.
        """
        task_description = f"Debug the following code snippet.\n{code_snippet}"
        prompt = self.prompt_manager.generate_code_writing_prompt(task_description)
        debugged_code = self.language_model.ask(prompt)
        return debugged_code


class CodeReviewer:
    def __init__(self, language_model):
        """
        Initialize the CodeReviewer with a language model and a prompt manager.

        :param language_model: An instance of a language model (e.g., LangModel, StructuredLangModel).
        :param prompt_manager: An instance of a prompt manager.
        """
        self.language_model = language_model
        self.prompt_manager = PromptManager(LangModel())

    def review_code(self, code_snippet):
        """
        Review the given code snippet for correctness, efficiency, and style.

        :param code_snippet: The code snippet to be reviewed.
        :return: The review feedback.
        """
        prompt = self.prompt_manager.generate_code_review_prompt(code_snippet)
        review_feedback = self.language_model.ask(prompt)
        return review_feedback

    def optimize_prompt(self, prompt, optimization_target):
        """
        Optimize the given prompt for the specified optimization target.

        :param prompt: The prompt to be optimized.
        :param optimization_target: The target for optimization (e.g., code quality, efficiency).
        :return: An optimized prompt.
        """
        optimized_prompt = self.prompt_manager.optimize_prompt(prompt, optimization_target)
        return optimized_prompt


 class PromptManager:
     def __init__(self, language_model):
         """
         Initialize the PromptManager with a language model.

         :param language_model: An instance of a language model (e.g., LangModel, StructuredLangModel).
         """
         self.language_model = language_model
         self.base_code_prompt = "You are an experienced Python coder. Your job is to write correct, efficient, and well-structured code to solve data-science problems.\n"
         self.base_code_review_prompt = "You are a senior data scientist. Your job is to review the code written by a junior data scientist for correctness, efficiency, and style.\n"

     def generate_code_writing_prompt(self, task_description):
         """
         Generate a prompt for code writing based on the task description.

         :param task_description: A description of the coding task.
         :return: A prompt for code writing.
         """
         prompt = self.base_code_prompt+ f"Write code to accomplish the following task: {task_description}"
         return prompt

     def generate_code_review_prompt(self, code_snippet):
         """
         Generate a prompt for code review based on the provided code snippet.

         :param code_snippet: A snippet of code to be reviewed.
         :return: A prompt for code review.
         """
         prompt = self.base_code_review_prompt + f"Review the following code for correctness, efficiency, and style: {code_snippet}"
         return prompt

     def optimize_prompt(self, prompt, optimization_target):
         """
         Optimize the given prompt using the language model. for the given optimization target.

         :param prompt: The prompt to be optimized.
          :param optimization_target:  The target for optimization (e.g., code quality, efficiency).
         :return: An optimized prompt.
         """
         prompt = f"Please optimize the prompt below so that the LLM output will be better with respect to {optimization_target}:\n'{prompt}'"
         optimized_prompt = self.language_model.ask(prompt)
         return optimized_prompt