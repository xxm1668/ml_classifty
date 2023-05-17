list1 = [2, 3, 4, 5]
list2 = [0, 0, 1, 1]

list3 = zip(list1, list2)
for l in list3:
    print(l)
print(list3)


def huiwenshu(str, index):
    low = index
    high = index
    while high < len(str) and str[low] == str[high]:
        high += 1
    high2 = high - 1
    while low >= 0 and high2 < len(str) and str[low] == str[high2]:
        low -= 1
        high2 += 1
    low += 1
    high2 -= 1
    return low, high2 + 1


def longestPalindrome(s):
    def huiwenshu(str, index):
        low = index
        high = index
        while high < len(str) and str[low] == str[high]:
            high += 1
        high2 = high - 1
        while low >= 0 and high2 < len(str) and str[low] == str[high2]:
            low -= 1
            high2 += 1
        low += 1
        high2 -= 1
        return low, high2 + 1

    max_len = 0
    start = 0
    end = 0
    for i in range(0, len(s)):
        _start, _end = huiwenshu(s, i)
        if _end - _start > max_len:
            max_len = _end - _start
            start = _start
            end = _end
    return s[start:end]


def numberOfArithmeticSlices(nums):
    res = 0
    dp = [0] * len(nums)

    for i in range(2, len(nums)):
        if nums[i] - nums[i - 1] == nums[i - 1] - nums[i - 2]:
            dp[i] = dp[i - 1] + 1
            res += dp[i]
    return res


def numDecodings(s):
    dp = [0] * len(s)
    s = '00' + s
    dp = [1, 1] + dp
    for i in range(2, len(s)):
        if 10 <= int(s[i - 1:i + 1]) <= 26:
            dp[i] += dp[i - 2]
        if s[i] != '0':
            dp[i] += dp[i - 1]
    return dp


def wordBreak2(s, wordDict):
    dp = [0] * len(s)
    _start = 0
    for i in range(len(s)):
        _start = i
        if s[0:i] != '' and s[0:i] not in wordDict:
            continue
        for j in range(i + 1, len(s) + 1):
            tmp = s[_start:j]
            if tmp in wordDict:
                dp[j - 1] = 1
                _start = j
        if _start == len(s):
            return True
    return False


def wordBreak(s, wordDict):
    dp = [False] * len(s)
    dp = [True] + dp
    for i in range(1, len(s) + 1):
        for j in range(i):
            tmp = s[j:i]
            if tmp in wordDict and dp[j]:
                dp[i] = True

    return dp[len(s)]


def jump(nums):
    if len(nums) == 1:
        return 0
    next_step = 0
    current_step = 0
    count = 0
    for i in range(len(nums)):
        next_step = max(nums[i] + i, next_step)
        if i == current_step:
            current_step = next_step
            count += 1
        if current_step >= len(nums) - 1:
            break
    return count


def rob(nums):
    def r(nums):
        if len(nums) == 0:
            return 0
        if len(nums) == 1:
            return nums[0]
        if len(nums) == 2:
            return max(nums)
        nums[2] += nums[0]
        for i in range(3, len(nums)):
            nums[i] += max(nums[i - 2], nums[i - 3])
        return max(nums)

    if len(nums) == 0:
        return 0
    if len(nums) == 1:
        return nums[0]
    if len(nums) == 2:
        return max(nums)
    return max(r(nums[0:len(nums) - 1]), r(nums[1:]))


def lengthOfLIS(nums):
    dp = [1] * len(nums)
    for i in range(1, len(nums)):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)
    print(dp)
    return max(dp)


def uniquePaths(m, n):
    if m == 1 and n == 1:
        return 1
    dp = []
    dp.append([0] + [1] * (m - 1))
    for i in range(1, n):
        dp.append([1] + [0] * (m - 1))
    dp[0][0] = 0
    for i in range(1, n):
        for j in range(1, m):
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
    return dp[n - 1][m - 1]


def canJump(nums):
    dp = [False] * len(nums)
    current_step = 0
    next_step = 0
    for i in range(len(nums)):
        next_step = max(nums[i] + i, next_step)
        if i == current_step:
            current_step = next_step
            dp[i] = True
        if current_step >= len(nums) - 1:
            dp[i] = True
    return dp[-1]


def longestCommonSubsequence(text1, text2):
    dp = []
    for i in range(len(text2) + 1):
        tmp = [0] * (len(text1) + 1)
        dp.append(tmp)
    for i in range(1, len(text1) + 1):
        for j in range(1, len(text2) + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    print(dp)
    return dp[len(text1)][len(text2)]


def minDistance(word1, word2):
    dp = []
    for i in range(len(word1) + 1):
        tmp = [0] * (len(word2) + 1)
        dp.append(tmp)
    for i in range(len(word1) + 1):
        dp[i][0] = i
    for j in range(len(word2) + 1):
        dp[0][j] = j
    for i in range(1, len(word1) + 1):
        for j in range(1, len(word2) + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + 1
    return dp[len(word1)][len(word2)]


def letterCombinations(digits):
    letters = {2: 'abc', 3: 'def', 4: 'ghi', 5: 'jkl', 6: 'mno', 7: 'pqrs', 8: 'tuv', 9: 'wxyz'}

    def combinations(combines, strs):
        result = []
        for i in strs:
            for combine in combines:
                result.append(combine + i)
        return result

    combines = []
    if len(digits) == 0:
        return combines
    letter = letters[int(digits[0])]
    for l in letter:
        combines.append(l)
    for i in range(len(digits)):
        if i == 0:
            continue
        letter = letters[int(digits[i])]
        combines = combinations(combines, letter)
    return combines


def generateParenthesis(n):
    if n <= 0:
        return []
    if n == 1:
        return ['()']

    def combine_str(combines):
        result = []
        for combine in combines:
            tmp = '()' + combine
            if tmp not in result:
                result.append(tmp)
            tmp = combine + '()'
            if tmp not in result:
                result.append(tmp)

            tmp = '(' + combine + ')'
            if tmp not in result:
                result.append(tmp)
        return result

    combines = ['()']
    for i in range(1, n):
        combines = combine_str(combines)
    return combines


def coinChange(coins, amount):
    if amount == 0:
        return 0
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for i in range(1, amount + 1):
        tmp = []
        for c in coins:
            if i - c >= 0:
                tmp.append(dp[i - c])
            else:
                tmp.append(float('inf'))
        dp[i] = min(tmp) + 1
    if dp[-1] == float('inf'):
        result = -1
    else:
        result = dp[-1]
    return result


def integerBreak(n):
    dp = [0] * (n + 1)
    dp[2] = 1
    for i in range(3, n + 1):
        for j in range(1, n):
            dp[i] = max(dp[i], dp[i - j] * j, (i - j) * j)
    print(dp)
    return dp[n]


if __name__ == '__main__':
    print(longestPalindrome('a'))
    ll = [1, 2]
    result = []
    res = numberOfArithmeticSlices([1, 2, 3, 4, 6])
    print(res)
    print(integerBreak(4))
