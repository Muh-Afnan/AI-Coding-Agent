from functions.get_file_content import get_file_content
from functions.write_file import write_file

def main():
    print(get_file_content("calculator", "lorem.txt"))
    print (write_file("calculator", "lorem.txt", content="This is content"))


main()