while True:
    string = input()
    if string == '.': break
    st = []
    flag = True
    for i in string:
        if i == '(':
            st.append(i)
        elif i == '[':
            st.append(i)
        elif i == ')':
            if st and st[-1] == '(':
                st.pop()
            else:
                flag = False
        elif i == ']':
            if st and st[-1] == '[':
                st.pop()
            else:
                flag = False
        
        if not flag: break
    
    if not st and flag:
        print('yes')
    else:
        print('no')    
    