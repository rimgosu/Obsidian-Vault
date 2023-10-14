# Deck Relationship Index

Using the deck (or deque) data structure, you can insert/delete data from both the front and the back.

At Alice University, during a data structures lecture, the deck was introduced. Now, Cheshire wants to use the deck to create a relationship between numbers.

Let's take a number A and split it by its digits and store it in a deck. We'll try to transform the values in the deck so they match the digits of another number, B.

If the minimum number of operations required to convert number A into number B using the deck is K, then we will refer to the relationship index between A and B as K.

To transform into B, the available operations are:

- Insert a single number either at the front or the back of the deck. (The number you can add is an integer between 0 and 9.)
- Delete a single number either at the front or the back of the deck.

For example, consider transforming the number 12345 into 245 using a deck:

|Step|Action|Explanation|
|---|---|---|
|0|{1,2,3,4,5}|Initial state.|
|1|{2,3,4,5}|Delete the value at the front.|
|2|{3,4,5}|Delete the value at the front.|
|3|{4,5}|Delete the value at the front.|
|4|{2,4,5}|Insert 2 at the front.|

As there are no more efficient steps, the relationship index between 12345 and 245 is 4.

Given two numbers, find their relationship index.

## Instructions

### Input

- The first line contains the number m, and the second line contains the number n.
    - The lengths of m and n are at least 500 and at most 3000.
    - The first digit of m and n can be 0.

### Output

- Output the relationship index between the two numbers.

## Example

### Input Example 1

Copy code

`12345 245`

### Output Example 1

Copy code

`4`

This is the same as the provided example.

### Input Example 2

Copy code

`111121 112111`

### Output Example 2

Copy code

`4`

Using the following actions, the conversion can be achieved in just 4 steps:

|Step|Action|Explanation|
|---|---|---|
|0|{1,1,1,1,2,1}|Initial state.|
|1|{1,1,1,2,1}|Delete the value at the front.|
|2|{1,1,1,2,1,1}|Insert 1 at the back.|
|3|{1,1,2,1,1}|Delete the value at the front.|
|4|{1,1,2,1,1,1}|Insert 1 at the back.|