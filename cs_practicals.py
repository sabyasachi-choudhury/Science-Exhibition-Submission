"""
Part 1: Getting total purchase bill
"""

model = input("""
    PLease select a model: 
    Hatchback: 5.35 lac
    Saloon: 4.95 lac
    Estate: 6.35 lac
    """).lower()
model_prices = {"hatchback": 535000, 'saloon': 495000, 'estate': 625000}
extra_prices = {'a': 45000, 'b': 5500, 'c': 10000, 'd': 350, 'e': 1000}
extra_reg = {'a': "Set of luxury seats",
             'b': "Satellite Navigation",
             "c": "Parking Sensors",
             "d": "Bluetooth Connectivity",
             "e": "Sound System"}
bill = {}
price = 0
price += model_prices.get(model, 0)
if price == 0:
    print("Please enter a correct model")
else:
    print(f"Current price: {price}")
    bill[model] = model_prices[model]
options = input("""
    Please enter the options for desired extras in the following format. Any other letters will not be registered - a,c,d.
    a) Set of Luxury seats - 45000
    b) Satellite Navigation - 55000
    c) Parking Sensors - 10000
    d) Bluetooth Connectivity - 350
    e) Sound System - 1000
    
    """).lower()

extras_price = 0
for char in options:
    if char in extra_reg.keys():
        extras_price += extra_prices.get(char, 0)
        bill[extra_reg[char]] = extra_prices[char]

price += extras_price

existing_customer = input("Are you an existing customer? Y/N: ").lower()
if existing_customer in ['y', 'yes']:
    bill['Existing Customer Discount'] = "10%"
else:
    bill['Existing Customer Discount'] = "0%"
trade_in = input("Are you trading in an old vehicle? Y/N: ").lower()
if trade_in in ['y', 'yes']:
    bill['trade in discount'] = "5%"
else:
    bill['trade in discount'] = "0%"

print("\n\tFinal Bill\n")
print("\t-Costs")
for key in list(bill.keys())[:-2]:
    print(f"{key}:{bill[key]}")
print("\n\t-Discounts")
discount = int(bill["Existing Customer Discount"][:-1]) + int(bill["trade in discount"][:-1])
print(f"Existing Customer Discount:{bill['Existing Customer Discount']}")
print(f"trade in discount:{bill['trade in discount']}")
print(f"net discount:{discount}%\n")
price *= (1 - discount * 0.01)
extras_price *= (1 - discount * 0.01)
print(f"\tTotal Price:{price}")
print()

"""
Part 2: Deciding on payment method
"""
print(f"""
    Let us look at your payment options now, sir/maam,
    
    A) Pay full amount now - You receive 1% cashback, or the optional extras free of charge. 
       Here, calculations assume cashback chosen. Pick A for further details
            Total paid: {price}
            Number of payments: 1
            Cashback: {0.01 * price}
            Overall cost: {0.99 * price}
    
    B) Pay in equal monthly installments over 4 years - no extra charge
            Total paid: {price}
            Number of payments: 48
            Amount per installment: {price / 48}
            Cashback: NA
            Overall cost: {price}
        
    C) Pay in equal monthly installments over 7 years - 5% extra charge
            Total paid: {price * 1.05}
            Number of payments: 84
            Amount per installment: {price * 1.05 / 84}
            Cashback: NA
            Overall cost: {price * 1.05}
""")

payment = input("Please pick chosen option: A, B, or C:\n").lower()
if payment not in 'abc':
    print("Please pick a valid choice")

if payment == 'a':
    cashback = 0.01 * price
    if cashback < extras_price:
        print(
            f"""
    Cashback is the better alternative
            Total paid: {price}
            Number of payments: 1
            Cashback: {0.01 * price}
            Overall cost: {0.99 * price}

    Extras free
            Total paid: {price - extras_price}:
            Number of payments: 1
            Cashback: NA
            Overall cost: {price - extras_price}
""")

    else:
        print(
            f"""
    Extras free
            Total paid: {price - extras_price}:
            Number of payments: 1
            Cashback: NA
            Overall cost: {price - extras_price}
            
    Cashback is the better alternative
            Total paid: {price}
            Number of payments: 1
            Cashback: {0.01 * price}
            Overall cost: {0.99 * price}
""")

    choice = input("Please pick between option a) cashback, or option b) extras:\n").lower()
    if choice == 'a':
        print("Thank you for your purchase! Your final receipt is")
        print("\t-Costs")
        for key in list(bill.keys())[:-2]:
            print(f"{key}:{bill[key]}")
        print("\n\t-Discounts")
        print(f"Existing Customer Discount:{bill['Existing Customer Discount']}")
        print(f"trade in discount:{bill['trade in discount']}")
        print(f"net discount:{discount}%\n")
        print(f"\tTotal Price before payment:{price}")
        print()
        print("Method of payment: All at once, with cashback")
        print(f"\tCashback: {cashback}")
        print(f"\n\tTotal Cost: {price - cashback}")
    elif choice == 'b':
        print("Thank you for your purchase! Your final receipt is")
        print("\t-Costs")
        for key in list(bill.keys())[:-2]:
            print(f"{key}:{bill[key]}")
        print("\n\t-Discounts")
        print(f"Existing Customer Discount:{bill['Existing Customer Discount']}")
        print(f"trade in discount:{bill['trade in discount']}")
        print(f"net discount:{discount}%\n")
        print(f"\tTotal Price before payment:{price}")
        print()
        print("Method of payment: All at once, with extras free")
        print(f"\tFree cost: {extras_price}")
        print(f"\n\tTotal Cost: {price - extras_price}")
    else:
        print("Please pick valid option")

elif payment == 'b':
    print("Thank you for your purchase! Your final receipt is")
    print("\t-Costs")
    for key in list(bill.keys())[:-2]:
        print(f"{key}:{bill[key]}")
    print("\n\t-Discounts")
    print(f"Existing Customer Discount:{bill['Existing Customer Discount']}")
    print(f"trade in discount:{bill['trade in discount']}")
    print(f"net discount:{discount}%\n")
    print(f"\tTotal Price before payment:{price}")
    print()
    print("Method of payment: Monthly installments over 4 years, no extra charge")
    print("Num installments: 48")
    print(f"Price per installment: {price / 48}")
    print(f"Total cost: {price}")

else:
    print("Thank you for your purchase! Your final receipt is")
    print("\t-Costs")
    for key in list(bill.keys())[:-2]:
        print(f"{key}:{bill[key]}")
    print("\n\t-Discounts")
    print(f"Existing Customer Discount:{bill['Existing Customer Discount']}")
    print(f"trade in discount:{bill['trade in discount']}")
    print(f"net discount:{discount}%\n")
    print(f"\tTotal Price before payment:{price}")
    print()
    print("Method of payment: Monthly installments over 7 years, 5% extra charge")
    print("Num installments: 84")
    print(f"Price per installment: {price * 1.05 / 84}")
    print(f"Total cost: {price * 1.05}")

import math


def primes(n):
    l = [2]
    k = 3
    while k <= n:
        prime = True
        for p in l:
            if p > math.sqrt(k):
                break
            if k % p == 0:
                prime = False
                break
        if prime:
            l.append(k)
        k += 1
    return l


def d_check(n_lim):
    p_list = primes(n_lim ** 2)[::-1]
    for n in range(3, n_lim):
        d = 0
        for p in p_list:
            if p < n ** 2 < p ** 2:
                if (n ** 2 // p) % 2 == 1:
                    d += 1
        if d <= n:
            print(n)

d_check(5000)