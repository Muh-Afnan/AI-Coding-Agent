import os
from google.genai import types

def get_files_info(working_directory, directory=None):
    # if directory and os.path.isabs(directory):
    #     target_dir = directory
    # else:
    #     target_dir = os.path.join(working_directory,directory) 

    # if not os.path.isdir(target_dir):
    #     return f'Error: "{directory}" is not a directory'
    
    abs_working_dir = os.path.abspath(working_directory)
    abs_directory = ""
    if directory is None:
        abs_directory = os.path.abspath(working_directory)
    else:
        abs_directory = os.path.abspath(os.path.join(working_directory,directory))
    
    if not abs_directory.startswith(abs_working_dir):
        return f'Error: Cannot list "{directory}" as it is outside the permitted working directory'
    
    if not os.path.isdir(working_directory):
        return f'Error: "{directory}" is not a directory'
 
    final_response = ""
    files = os.listdir(abs_directory)
    for file in files:
        path = os.path.join(abs_directory,file)
        size = os.path.getsize(path)
        vaild_file = os.path.isdir(path)
        final_response += f"- {file}:file_size={size} bytes, is_dir={vaild_file}\n"
    return final_response


schema_get_files_info = types.FunctionDeclaration(
    name="get_files_info",
    description="Lists files in the specified directory along with their sizes, constrained to the working directory.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "directory": types.Schema(
                type=types.Type.STRING,
                description="The directory to list files from, relative to the working directory. If not provided, lists files in the working directory itself.",
            ),
        },
    ),
)

