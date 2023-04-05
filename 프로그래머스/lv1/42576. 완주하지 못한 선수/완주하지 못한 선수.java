import java.util.HashMap;
class Solution {
    public String solution(String[] participant, String[] completion) {
        String answer = "";
        HashMap<String, Integer> hashMap = new HashMap<String, Integer>();


        for (String e : participant) {
            if (!hashMap.containsKey(e)) {
                hashMap.put(e, 0);
            }
            hashMap.put(e, hashMap.get(e) + 1);
        }

        for (String e : completion) {
            if (hashMap.containsKey(e)) {
                hashMap.put(e, hashMap.get(e) + 1);
            }
        }

        for (String e : participant) {
            if (hashMap.get(e) % 2 != 0) {
                answer = e;
            }
        }
        return answer;
    }
}