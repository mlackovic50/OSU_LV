#Zad 1.4.1.
'''
def total_euro(hours, rate):
    return hours * rate

hours = float(input("Broj radnih sati:"))
rate = float(input("Vaša satnica:"))
print(f"Zaradili ste {total_euro(hours,rate)} eura ovaj mjesec")
'''

#Zad 1.4.2.
'''
try:
    grade = float(input("Unesite broj između 0.1 i 1.0:"))
    if grade < 0.0 or grade > 1.0:
        print("Ocjena mora biti između 0.0 i 1.0")
    elif (grade < 0.6):
        print("F")
    elif (grade >= 0.6 and grade < 0.7):
        print("D")
    elif (grade >= 0.7 and grade < 0.8):
        print("C")
    elif (grade >= 0.8 and grade < 0.9):
        print("B")
    else:
        print("A")
except:
    print("Molim unesite broj")
'''
#Zad 1.4.3.
'''
numbers = []
while True:
    user_input = input("Unesite broj ili 'Done' za kraj petlje:")
    if user_input.lower() == "done":
        break
    try:
        number = float(user_input)
        numbers.append(number)
    except ValueError:
        print("Please enter a numerical value")

if numbers:
    print(f"Brojeva uneseno: {len(numbers)}")
    print(f"Srednja vrijednost: {float(sum(numbers) / len(numbers))}")
    print(f"Minimalna vrijednost: {min(numbers)}")
    print(f"Maksimalna vrijednost: {max(numbers)}")
    print(f"Sortirana lista: {sorted(numbers)}")
else:
    print("Lista je prazna")
'''
#Zad 1.4.4.
'''
word_count = {}
try:
    with open("song.txt", "r", encoding="utf-8") as file:
        for line in file:
            words = line.split()
            for word in words:
                word = word.lower().strip(".,!?()[]{}:;\"'")
                word_count[word] = word_count.get(word, 0) + 1

    unique_words = [word for word, count in word_count.items() if count == 1]
    print(f"Broj riječi koje se pojavljuju samo jednom: {len(unique_words)}")
    print("Riječi koje se pojavljuju samo jednom:", unique_words)
except FileNotFoundError:
    print("Datoteka song.txt nije pronađena.")
'''
#Zad 1.4.5.
spam_word_count = 0
spam_exclamation_count = 0
ham_word_count = 0
spam_messages = 0
ham_messages = 0

try:
    with open("SMSSpamCollection.txt", "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            label, message = parts[0], parts[1]
            word_count = len(message.split())

            if label == "spam":
                spam_word_count += word_count
                spam_messages += 1
                if message.strip().endswith("!"):
                    spam_exclamation_count += 1
            elif label == "ham":
                ham_word_count += word_count
                ham_messages += 1

    if ham_messages > 0:
        print(f"Prosječan broj riječi u ham porukama: {ham_word_count / ham_messages:.2f}")
    if spam_messages > 0:
        print(f"Prosječan broj riječi u spam porukama: {spam_word_count / spam_messages:.2f}")
        print(f"Broj spam poruka koje završavaju uskličnikom: {spam_exclamation_count}")
except FileNotFoundError:
    print("Datoteka SMSSpamCollection.txt nije pronađena.")
except Exception as e:
    print(f"Došlo je do greške: {e}")
