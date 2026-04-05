import os
import time

def clear_screen():
    """Clears the console screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def draw_heart():
    """Draws a heart on the console."""
    heart = [
        "      ######      ######      ",
        "    ##########  ##########    ",
        "  ##########################  ",
        "  ##########################  ",
        "  ##########################  ",
        "    ######################    ",
        "      ##################      ",
        "        ##############        ",
        "          ##########          ",
        "            ######            ",
        "              ##              "
    ]

    clear_screen()
    print("\n\n\n")
    for line in heart:
        print(line.center(80))
        time.sleep(0.1)
    
    print("\n\n")
    print("A special heart for you!".center(80))
    print("\n\n")

if __name__ == "__main__":
    draw_heart()
