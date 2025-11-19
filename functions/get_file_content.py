import os
from config import MAX_CHARS
from google.genai import types



def get_file_content(working_directory, file_path):
    abs_working_dir = os.path.abspath(working_directory)
    file_path_complete = os.path.join(working_directory,file_path)
    abs_file_path = os.path.abspath(file_path_complete)
    if not abs_file_path.startswith(abs_working_dir):
        return f'Error: Cannot read "{abs_file_path}" as it is outside the permitted working directory'
    
    if not os.path.isfile(abs_file_path):
        return f'Error: File not found or is not a regular file: "{abs_file_path}"'
    
    file_content_string = ""
    try:
        with open(abs_file_path, "r") as f:
            file_content_string = f.read()
            if len(file_content_string)>MAX_CHARS:
                file_content_string = file_content_string[:MAX_CHARS] + f"... File {abs_file_path} truncated at {MAX_CHARS} characters"
        return file_content_string
    except Exception as e:
        return f"Exception Reading file :{e}"


schema_get_files_content = types.FunctionDeclaration(
    name="get_files_content",
    description="Get the the contents of the given file as a string, constrained to the working directory.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "file_path": types.Schema(
                type=types.Type.STRING,
                description="The path to the file, from working directory.",
            ),
        },
    ),
)