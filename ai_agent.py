# ai_agent.py
import os
import pandas as pd
import matplotlib.pyplot as plt
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Load environment variables (your API key)
load_dotenv()

class DataAnalysisAgent:
    """
    An AI agent that generates and executes Python code to analyze data.
    Designed for minimal API cost and maximum reliability.
    """
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the agent with the DataFrame it will analyze.
        """
        self.df = df
        # Initialize the LLM - using gpt-3.5-turbo for cost efficiency
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,  # Set to 0 for deterministic, reliable code generation
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.output_parser = StrOutputParser()

    def _create_system_prompt(self, context: str) -> str:
        """
        Creates a precise system prompt with the retrieved context and strict rules.
        This is key to minimizing errors and API cost.
        """
        return f"""You are a expert Python data analyst. Your task is to generate accurate, efficient pandas code to answer a user's question about a dataset.

CONTEXT FROM THE DATASET:
{context}

THE DATAFRAME:
- The DataFrame is already loaded as `df`.
- Do not create a new DataFrame or read from a file.
- You can assume `import pandas as pd` and `import matplotlib.pyplot as plt` are already done.

STRICT RULES:
1. Generate ONLY valid Python code. No explanations, no markdown code blocks.
2. The code must be efficient and use vectorized pandas operations.
3. If the question requires a plot, create it using `plt` and leave it open with `plt.show()`.
4. If the question is about the data structure (columns, dtypes), use `df.info()`, `df.columns`, etc.
5. For statistics, use `df.describe()`, `df['col'].mean()`, etc.
6. NEVER use methods or columns that are not in the context above.
7. DOUBLE-CHECK your code for typos before generating it.
8. Common methods: .tolist(), .mean(), .max(), .min(), .sum(), .unique()
9. If you cannot answer from the given context, generate: "# I need more context to answer this."

EXAMPLE QUESTIONS AND CODE:
Q: "What are the column names?"
A: df.columns.tolist()

Q: "What is the average price?"
A: df['Price'].mean()

Q: "Plot a histogram of prices"
A: plt.hist(df['Price'])
plt.title('Histogram of Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

Now, generate clean, accurate code for the following question:
"""

    def execute_code_safely(self, code: str):
        """
        Executes the generated Python code in a highly restricted environment.
        Returns the actual result of the computation, not the code itself.
        """
        # Clean the code first
        code = code.strip().strip('"').strip("'")
        
        # Remove markdown code blocks if present
        if code.startswith('```python'):
            code = code[9:]
        if code.startswith('```'):
            code = code[3:]
        if code.endswith('```'):
            code = code[:-3]
        code = code.strip()
        
        # Allow only safe imports and the existing DataFrame
        allowed_locals = {'df': self.df, 'pd': pd, 'plt': plt}
        allowed_globals = {}
        
        try:
            # First, try to evaluate it as an expression (e.g., "df['col'].min()")
            # This will return the actual result
            result = eval(code, allowed_globals, allowed_locals)
            return result
            
        except Exception as e:
            # If eval fails (e.g., for multi-line code or plots), try exec
            try:
                # For exec, we need to capture the result differently
                # Create a variable to store the result
                exec_vars = {}
                exec_vars.update(allowed_globals)
                exec_vars.update(allowed_locals)
                
                # Execute the code
                exec(code, exec_vars, exec_vars)
                
                # Try to get a result if it's a simple expression
                try:
                    # Check if the code was a simple expression
                    if len(code.split('\n')) == 1 and '=' not in code:
                        result = eval(code, exec_vars, exec_vars)
                        return result
                except:
                    pass
                
                # Check if a plot was created
                if 'plt' in exec_vars and exec_vars['plt'].get_fignums():
                    fig = exec_vars['plt'].gcf()
                    exec_vars['plt'].close()
                    return fig
                
                # If we can't get a specific result, return success message
                return "Code executed successfully (no return value)."
                
            except Exception as exec_error:
                return f"Error executing code: {str(exec_error)}"

    def generate_answer(self, question: str, context: str) -> str:
        """
        The main method: generates code, executes it, and formats a response.
        """
        print(f"🤖 Generating code for: '{question}'")
        
        # 1. Create the precise prompt
        system_prompt = self._create_system_prompt(context)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=question)
        ]
        
        # 2. Generate code with the LLM
        generated_code = self.llm.invoke(messages).content
        # Remove any surrounding quotes
        generated_code = generated_code.strip().strip('"').strip("'")
        print(f"   Generated code:\n{generated_code}")
        
        # 3. Execute the code safely
        execution_result = self.execute_code_safely(generated_code)
        print(f"   Execution result: {str(execution_result)[:100]}...")
        
        # 4. Format a final answer
        if "Error executing code" in str(execution_result):
            return f"I encountered an error: {execution_result}. Please try rephrasing your question."
        elif "I need more context" in generated_code:
            return "I couldn't find enough information in the data to answer this question. Please try asking about the data that's available."
        else:
            # Handle different types of results intelligently
            if isinstance(execution_result, (pd.DataFrame, pd.Series)):
                return f"Here are the results:\n\n{execution_result.to_string()}"
            elif hasattr(execution_result, '__iter__') and not isinstance(execution_result, str):
                return f"**Result:** {list(execution_result)}"
            else:
                return f"**Answer:** {execution_result}"

# Example function to tie everything together
def create_ai_agent(df: pd.DataFrame):
    """Factory function to create a new AI agent instance."""
    return DataAnalysisAgent(df)