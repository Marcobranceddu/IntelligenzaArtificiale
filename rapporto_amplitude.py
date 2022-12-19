import csv


# apre il file .csv
with open('augmented_amplitudes.csv') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]

# divide ogni elemento per 1024, per ottenere il rapporto dell'ampiezza in relazione alla dimensione dell'immagine
for row in range(len(rows)):
    for element in range(len(rows[row])):
        rows[row][element] = int(rows[row][element])/1024

# salva in rapporto_augmented_amplitudes.csv
with open('rapporto_augmented_amplitudes.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(rows)