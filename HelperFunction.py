def findPokerHand(hand):
    ranks = []
    suits = []
    possibleRanks = []
 
    for card in hand:
        if len(card) == 2:
            rank = card[0]
            suit = card[1]
        else:
            rank = card[0:2]
            suit = card[2]
        if rank == "A":
            rank = 14
        elif rank == "K":
            rank = 13
        elif rank == "Q":
            rank = 12
        elif rank == "J":
            rank = 11
        ranks.append(int(rank))
        suits.append(suit)
 
    sortedRanks = sorted(ranks)
 
    # Royal Flush and Straight Flush and Flush
    if suits.count(suits[0]) == 5: # Check for Flush Type
        if 14 in sortedRanks and 13 in sortedRanks and 12 in sortedRanks and 11 in sortedRanks \
                and 10 in sortedRanks:
            possibleRanks.append(1) # -- Royal Flush
        elif all(sortedRanks[i] == sortedRanks[i - 1] + 1 for i in range(1, len(sortedRanks))):
            possibleRanks.append(2) # -- Straight Flush
        else:
            possibleRanks.append(5) # -- Flush
 
    # Straight
    if all(sortedRanks[i] == sortedRanks[i - 1] + 1 for i in range(1, len(sortedRanks))):
        possibleRanks.append(6)
 
    handUniqueVals = list(set(sortedRanks))
 
    # Four of a kind and Full House
    if len(handUniqueVals) == 2:
        for val in handUniqueVals:
            if sortedRanks.count(val) == 4:  # --- Four of a kind
                possibleRanks.append(3)
            if sortedRanks.count(val) == 3:  # --- Full house
                possibleRanks.append(4)
 
    # Three of a Kind and Pair
    if len(handUniqueVals) == 3:
        for val in handUniqueVals:
            if sortedRanks.count(val) == 3:  # -- three of a kind
                possibleRanks.append(7)
            if sortedRanks.count(val) == 2:  # -- two pair
                possibleRanks.append(8)
 
    # Pair
    if len(handUniqueVals) == 4:
        possibleRanks.append(9)
 
    if not possibleRanks:
        possibleRanks.append(10)

    pokerHandRanks = {1: "Royal Flush", 2: "Straight Flush", 3: "Four of a Kind", 4: "Full House", 5: "Flush",
                      6: "Straight", 7: "Three of a Kind", 8: "Two Pair", 9: "Pair", 10: "High Card"}
    output = pokerHandRanks[min(possibleRanks)]

    print(hand, output)
    return output
 
 
if __name__ == "__main__":
    findPokerHand(["KH", "AH", "QH", "JH", "10H"])  # Royal Flush
    findPokerHand(["QC", "JC", "10C", "9C", "8C"])  # Straight Flush
    findPokerHand(["5C", "5S", "5H", "5D", "QH"])  # Four of a Kind
    findPokerHand(["2H", "2D", "2S", "10H", "10C"])  # Full House
    findPokerHand(["2D", "KD", "7D", "6D", "5D"])  # Flush
    findPokerHand(["JC", "10H", "9C", "8C", "7D"])  # Straight
    findPokerHand(["10H", "10C", "10D", "2D", "5S"])  # Three of a Kind
    findPokerHand(["KD", "KH", "5C", "5S", "6D"])  # Two Pair
    findPokerHand(["2D", "2S", "9C", "KD", "10C"])  # Pair
    findPokerHand(["KD", "5H", "2D", "10C", "JH"])  # High Card