import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

class Solution {
    public int[] solution(int[] array, int[][] commands) {
        ArrayList<Integer> answerList = new ArrayList<Integer>();
        int[] answer = new int[commands.length];

        for (int i = 0; i < commands.length; i++) {
            int i_ = commands[i][0];
            int j_ = commands[i][1];
            int k_ = commands[i][2];
            ArrayList<Integer> arrayList = new ArrayList<Integer>();

            for (int j = i_ - 1; j < j_; j++) {
                arrayList.add(array[j]);
            }

            Collections.sort(arrayList);
            answerList.add(arrayList.get(k_ - 1));
            answer[i] = answerList.get(i).intValue();
        }
        return answer;
        }
}