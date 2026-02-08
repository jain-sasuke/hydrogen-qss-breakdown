c    *  ba5.f: Program for computing hydrogenic radial integrals,
c    *  ocillator strengths, Einstein coefficients.
c    *  nu= principal quantum number (n) of upper state
c    *  nl= principal quantum number n') of lower state
c    *  lu= orbital quantum number (l) of upper state
c    *  ll= orbital quantum number (l') of lower state
c    *  z= Z= nuclear charge
c    *  am= M= nuclear mass in atomic units
c    *  ain= R(n,l;n'l')=radial integral
c    *  ain2= ain**2
c    *  os= f(n',l';n,l)= absorption oscillator strength
c    *  ein= A(n,l;n',l')= Einstein coefficient
c    *  subroutine fb5z(z,am,nl,nu,lu,ain,ain2,os,ein)
c    *  subroutine fa5z(z,am,nl,nu,lu,ain,ain2,os,ein)
c
        iunit=5
        iread=2
        open(file='ba5.out',unit=iunit)
        open(file='ba5.in',unit=iread)
c
1        format(2x,' lu     ','ll    ',
     2        'R**2',9x,'f(nl,ll;nu,lu)',3x,'A(nu,lu;nl,ll)',x//)
2        format('ba5.out'//)
3        format('Z=', 1pe11.4, 10x, ' M=', 1pe11.4)
4        format(/)
5        format(' nu='i4,2x,'nl=',i4,3x, 2(1pe11.4,3x))
6        format (3(i4,3x))
55        format (2(i4,3x), 2(1pe11.4,3x),3x,1pe11.4,3x)
c
        read (iread,*) nu,nl,z,am
c
        write(iunit,2)
        write(iunit,3) z,am
        write(iunit,5) nu,nl
        write(iunit,4)
        write(iunit,1)
c
        luf=nu-1
c
        do 34 iu=1,luf
        lu=iu
        ll=lu-1
        call fb5z(z,am,nl,nu,lu,ain,ain2,os,ein)
        write(iunit,55) lu,ll,ain2,os,ein
34        continue
        write(iunit,4)
c
        llf= nl-1
        do 33 ll = 1, llf
        lu=ll -1
        call fa5z(z,am,nl,nu,lu,ain,ain2,os,ein)
        write(iunit,55) lu,ll,ain2,os,ein
        lu=ll+1
        if(lu .gt. ll) go to 33
33        continue
c
        stop
        end
c
        subroutine fb5z(z,am,nl,nu,lu,ain,ain2,os,ein)
c    *        recurrence on b
c
        dimension h(1100)
c
        n=nu
        np=nl
        ll=lu-1
        lp=ll
c
        cl=2.9979250e+10
        ryi=109737.312
        em=5.4859e-04
        rkay=ryi/(1.+em/am)
c
        l=lp+1
        a01=-n+l+1.
        a02=a01-2.
        c0=2.*l
c
        b=a01
        e1=0.
        c = c0
        x=-4.*n*np/(n-np)**2
c
        h(1)=1.
        h(2)= 1.-b*x/c
        i1=(np-l)
        if(i1 .eq. 0) go to 40
        if(i1 .eq. 1) go to 41
c
        do 4 i=2,i1
        j=i+1
        a=-i+1.
        h1=-a*(1.-x)*h(j-2) /(a-c)
        h2=(a*(1.-x)+(a+b*x-c))*h(j-1)/(a-c)
        h(j)= h1+h2
        if(abs(h(j)) .gt. 1.e+25) go to 30
        go to 4
c
30        continue
        h(j)=h(j)/1.e+25
        h(j-1)=h(j-1)/1.e+25
        e1=e1+25.
c
4        continue
        p1=h(i1+1)
        go to 50
c
40        continue
        p1=h(1)
        go to 50
c
41        continue
        p1=h(2)
        go to 50
c
50        continue
        b=a02
        e2=0.
        h(1)=1.
        h(2)= 1.-b*x/c
        i1=(np-l)
        if(i1 .eq. 0) go to 42
        if(i1 .eq. 1) go to 43
c
        do 5 i=2,i1
        j=i+1
        a=-i+1.
        h1=-a*(1.-x)*h(j-2) /(a-c)
        h2=(a*(1.-x)+(a+b*x-c))*h(j-1)/(a-c)
        h(j)= h1+h2
        if(abs(h(j)) .gt. 1.e+25) go to 31
        go to 5
c
31        continue
        h(j)=h(j)/1.e+25
        h(j-1)=h(j-1)/1.e+25
        e2=e2+25.
c
5        continue
        p2=h(i1+1)
        go to 51
c
42        continue
        p2=h(1)
        go to 51
c
43        continue
        p2=h(2)
        go to 51
c
51        continue
        cc4=n-np
        cc5=n+np
        ff= p1*(1.-cc4**2/cc5**2*p2/p1*10.**(e2-e1))
        alof=alog10(abs(ff))+e1
c
c    *        cal of c1, c2, c3, c4, c5
c
        i2=(2*l-1)
        s1=0.
c
        do 6 i=1,i2
        ai=i
        s1=s1+ alog10(ai)
6        continue
c
        c1= - (alog10(4.)+s1)
c
        s=0.
c
        i3=(n+l)
        si3=0.
c
        do 7 i=1,i3
        ai=i
        si3=si3+ alog10(ai)
7        continue
c
        s=s+si3
c
        i4=(np+l-1)
        si4=0.
c
        do8 i=1,i4
        ai=i
        si4=si4+ alog10(ai)
8        continue
c
        s=s+si4
c
        i5=n-l-1
        si5=0.
        if(i5 .eq. 0) go to 2
c
        do 9i=1,i5
        ai=i
        si5=si5+ alog10(ai)
9        continue
c
        s=s-si5
c
2        continue
        i6=i1
        si6=0.
        if(i6 .eq. 0)go to 3
c
        do 12i=1,i6
        ai=i
        si6=si6+ alog10(ai)
12        continue
c
         s=s-si6
c
3        continue
        c2=s/2.
c
        cc3=4.*n*np
        c3= (l+1.)*alog10(cc3)
c
        cc4=n-np
        c4= (n+np-2.*l-2.)*alog10(cc4)
c
        cc5=n+np
        c5=(-n-np)*alog10(cc5)
c
        c=c1+c2+c3+c4+c5
c
        ali =alof+c
        ain = 10.**ali
        ain2=ain**2
c
        xu=nu
        xl=nl
        euplo=rkay*z**2/xu**2/xl**2*(xu-xl)*(xu+xl)
        os=1./3.*(xu+xl)*(xu-xl)/(xu*xl)**2*max(lu,ll)/
     2        (2.*ll+1.)*ain2*z**2
        ein=0.66704*(2.*ll+1.)/(2.*lu+1.)*(euplo)**2*os
        return
        end
c
        subroutine fa5z(z,am,nl,nu,lu,ain,ain2,os,ein)
c    *        recurrence on a
c
        dimension h(1100)
        cl=2.9979250e+10
        ryi=109737.312
        em=5.4859e-04
        rkay=ryi/(1.+em/am)
c
        n=nl
        ll=lu+1
        l=ll
        b0= 0.-nu+l+0.
        c0=2.*l
c
        b=b0
        c = c0
        cc4=1.*(n-nu)
        x=-4.*n*nu/cc4**2
c
        h(1)=1.
        h(2)= 1.-b*x/c
        e1=0.
        e2=e1
        i1=(n-l+1)
c
        do 4 i=2,i1
        j=i+1
        a=1.-i+0.
        h1=-a*(1.-x)*h(j-2) /(a-c)
        h2=(a*(1.-x)+(a+b*x-c))*h(j-1)/(a-c)
        h(j)= h1+h2
        if(abs(h(j)) .gt. 1.e+25) go to 30
        go to 4
c
30        continue
        h(j)=h(j)/1.e+25
        h(j-1)=h(j-1)/1.e+25
        h(j-2)=h(j-2)/1.e+25
        e1=e1+25.
        e2=e1
c
4        continue
        p1=h(i1-1)
        p2 = h(i1+1)
c
        cc4=1.*(n-nu)
        cc4= abs(cc4)
        cc5=n+nu
        ff= p1*(1.-cc4**2/cc5**2*p2/p1*10.**(e2-e1))
        alof=alog10(abs(ff))+e1
c
        i2=(2*l-1)
        s1=0.
c
        do 6 i=1,i2
        ai=i
        s1=s1+ alog10(ai)
6        continue
c
        c1= - (alog10(4.)+s1)
         s=0.
c
        i3=(n+l)
        si3=0.
c
        do 7 i=1,i3
        ai=i
        si3=si3+ alog10(ai)
7        continue
c
        s=s+si3
c
        i4=(nu+l-1)
        si4=0.
c
        do8 i=1,i4
        ai=i
        si4=si4+ alog10(ai)
8        continue
c
        s=s+si4
c
        i5=n-l-1
        si5=0.
        if(i5 .eq. 0) go to 2
c
        do 9i=1,i5
        ai=i
        si5=si5+ alog10(ai)
9        continue
c
        s=s-si5
c
2        continue
c
        i6=nu-l
        si6=0.
        if(i6 .eq. 0) go to 3
c
        do 12 i=1,i6
        ai=i
        si6=si6+ alog10(ai)
12        continue
c
        s=s-si6
c
3        continue
        c2=s/2.
c
        cc3=4.*n*nu
        c3= (l+1.)*alog10(cc3)
c
        cc4=cc4
        c4= (n+nu-2.*l-2.)*alog10(cc4)
c
        cc5=n+nu
        c5=(-n-nu)*alog10(cc5)
c
        c=c1+c2+c3+c4+c5
c
        ali =alof+c
        ain = 10.**ali/z
        ain2=ain**2
c
        xu=nu
        xl=nl
        euplo=rkay*z**2/xu**2/xl**2*(xu-xl)*(xu+xl)
        os=1./3.*(xu+xl)*(xu-xl)/(xu*xl)**2*max(lu,ll)/
     2        (2.*ll+1.)*ain2*z**2
        ein=0.66704*(2.*ll+1.)/(2.*lu+1.)*(euplo)**2*os
c
        return
        end