import os
import subprocess
from google.genai import types

def run_python_file(working_directory, file_path, args=[]):
    abs_working_dir = os.path.abspath(working_directory)
    file_path_complete = os.path.join(working_directory,file_path)
    abs_file_path = os.path.abspath(file_path_complete)

    if not abs_file_path.startswith(abs_working_dir):
        return f'Error: Cannot execute "{file_path}" as it is outside the permitted working directory'
    if not os.path.isfile(abs_file_path):
        return f'Error: File "{file_path}" not found.'
    if not abs_file_path.endswith(".py"):
        return f'Error: "{file_path}" is not a Python file.'
    
    try:
        final_arg = ['python',file_path]
        final_arg.extend(args)
        output = subprocess.run(final_arg, cwd=abs_working_dir,timeout=30,capture_output=True)
        print(output)
        final_string =  f"""
            STDOUT: {output.stdout}
            STDERR: {output.stderr}
        """
        if output.stdout == "" and output.stderr== "":
            final_string = "No output produced.\n"
        if output.returncode!=0:
            final_string += f"Process exited with code {output.returncode}"
        return final_string
    except Exception as e:
        return e
    
    return True

schema_run_python_fle = types.FunctionDeclaration(
    name="run_python_fle",
    description="Run python file with python interpreter. Accepts additional CLI args as an optional array.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "file_path": types.Schema(
                type=types.Type.STRING,
                description="The file to run, relative to the working directory."
            ),
            "args": types.Schema(
                type=types.Type.ARRAY,
                description="An optional array of strings to be used as the CLI args for the Python file",
                items= types.Schema(type=types.Type.STRING,),
            ),
        },
    ),
)