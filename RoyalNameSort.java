import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/**
 * RoyalNameSort
 */
public class RoyalNameSort {
    

    public static void main(String[] args) {
        List<String> name_list = new ArrayList();
        name_list.add("James II");
        name_list.add("James I");
        name_list.add("James V");
        name_list.add("Williams I");
        name_list.add("Geoge I");

        Collections.sort(name_list,new Comparator<String>() {
            @Override
            public int compare(String o1, String o2) {
                String[] o1_array = o1.split(" ");
                String[] o2_array = o2.split(" ");
                if(o1_array[0].compareTo(o2_array[0])>0){
                    return 1;
                }else if (o1_array[0].compareTo(o2_array[0])< 0){
                    return -1;
                }else{
                    return compare_roman_num(o1_array[1], o2_array[1]);
                }
            }
        });

        System.out.println(name_list);
    }

    public static int compare_roman_num(String s1, String s2) {
        List<String> ROMAN_NUMS = new ArrayList();
        ROMAN_NUMS.add("I");
        ROMAN_NUMS.add("II");
        ROMAN_NUMS.add("III");
        ROMAN_NUMS.add("V");

        int result = ROMAN_NUMS.indexOf(s1) -ROMAN_NUMS.indexOf(s2);
        if(result>0){
            return 1;
        }
        if(result<0){
            return -1;
        }
        return result;
        
    }
}