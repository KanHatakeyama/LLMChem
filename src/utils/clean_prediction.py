import re

def extract_numbers(test_strings):
    """
    TODO:

    一部で変換できてない
    For "168-170°C", it extracted and converted to [168, 170].
    For "-150", it could not extract a number, resulting in an empty list [].
    For "345", it extracted and converted to [345].
    For "-10 to 0", it extracted and converted to [-10, 0].
    For "-234 to -100", it extracted and converted to [-234, -100].
    """

    # Regular expression pattern to match numbers and 'number to number' or 'number-number' patterns
    pattern = r"-?\d+(?:\s*to\s*-?\d+|-\d+)?"

    # Applying the regular expression to each test string
    comprehensive_matches = [re.findall(pattern, test_string) for test_string in test_strings]

    # Function to process the matches and convert them into integer lists
    def convert_matches(matches):
        converted_result = []
        for match in matches:
            # Splitting the string if it contains 'to' or '-'
            if ' to ' in match:
                numbers = match.split(' to ')
            elif '-' in match and match.count('-') == 1:
                numbers = match.split('-')
            else:
                numbers = [match]

            # Converting the split strings to integers
            try:
                converted_numbers = list(map(int, numbers))
                converted_result.append(converted_numbers)
            except ValueError:
                # If conversion fails, append an empty list
                converted_result.append([])

        return converted_result

    # Processing the comprehensive matches
    final_converted_results = [convert_matches(match) for match in comprehensive_matches]
    return final_converted_results