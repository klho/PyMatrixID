c
c
c       dependencies: prini, idz_house, idz_qrpiv, idz_id, id_rand,
c                     idzp_rid, idz_id2svd, lapack.a, blas.a
c
c
        implicit none
c
        integer len
        parameter(len = 1 000 000)
c
        integer m,n,krank,ier,iu,iv,is,lw,k
        real*8 eps,errmax,errrms,s(100 000)
        complex*16 a(len),b(len),dummy,w(len),u(len),v(len)
        external matveca,matvec
c
c
        call prini(6,13)
c
c
        print *,'Enter m:'
        read *,m
        call prinf('m = *',m,1)
c
        print *,'Enter n:'
        read *,n
        call prinf('n = *',n,1)
c
        krank = 5
        call prinf('krank = *',krank,1)
c
c
c       Fill a.
c
        call fill(krank,m,n,a,s)
        call prin2('a = *',a,m*n)
        call prin2('s = *',s,krank+1)
c
c
c       Calculate an SVD approximating a.
c
        eps = .1d-11
        lw = len
c
        call idzp_rsvd(lw,eps,m,n,matveca,a,dummy,dummy,dummy,
     1                 matvec,a,dummy,dummy,dummy,krank,
     2                 iu,iv,is,w,ier)
c
        call prinf('ier = *',ier,1)
c
c
c       Copy u, v, and s from w.
c
        do k = 1,krank*m
          u(k) = w(iu+k-1)
        enddo ! k
c
        do k = 1,krank*n
          v(k) = w(iv+k-1)
        enddo ! k
c
        do k = 1,krank
          s(k) = w(is+k-1)
        enddo ! k
c
c
c       Construct b = u diag(s) v^*.
c
        call zreconsvd(m,krank,u,s,n,v,b)
c
c
c       Compute the difference between a and b.
c
        call materr(m,n,a,b,errmax,errrms)
        call prin2('errmax = *',errmax,1)
        call prin2('errrms = *',errrms,1)
c
c
        stop
        end
c
c
c
c
        subroutine fill(krank,m,n,a,s)
c
c       fills an m x n matrix with suitably decaying singular values,
c       and left and right singular vectors taken from the DFT.
c
c       input:
c       krank -- one less than the rank of the matrix to be constructed
c       m -- first dimension of a
c       n -- second dimension of a
c
c       output:
c       a -- filled matrix
c       s -- singular values of a
c
        implicit none
        integer krank,j,k,l,m,n
        real*8 r1,pi,s(krank+1)
        complex*16 a(m,n),sum,ci
c
        r1 = 1
        pi = 4*atan(r1)
        ci = (0,1)
c
c
c       Specify the singular values.
c
        do k = 1,krank
          s(k) = exp(log(1d-10)*(k-1)/(krank-1))
        enddo ! k
c
        s(krank+1) = 1d-10
c
c
c       Construct a.
c
        do k = 1,n
          do j = 1,m
c
            sum = 0
c
            do l = 1,krank
              sum = sum+exp(2*pi*ci*(j-r1)*(l-r1)/m)*sqrt(r1/m)
     1                 *exp(2*pi*ci*(k-r1)*(l-r1)/n)*sqrt(r1/n)*s(l)
            enddo ! l
c
            l = krank+1
            sum = sum+exp(2*pi*ci*(j-r1)*(l-r1)/m)*sqrt(r1/m)
     1               *exp(2*pi*ci*(k-r1)*(l-r1)/n)*sqrt(r1/n)*s(l)
c
            a(j,k) = sum
c
          enddo ! j
        enddo ! k
c
c
        return
        end
c
c
c
c
        subroutine materr(m,n,a,b,errmax,errrms)
c
c       calculates the relative maximum and root-mean-square errors
c       corresponding to how much a and b differ.
c
c       input:
c       m -- first dimension of a and b
c       n -- second dimension of a and b
c       a -- matrix whose difference from b will be measured
c       b -- matrix whose difference from a will be measured
c
c       output:
c       errmax -- ratio of the maximum elementwise absolute difference
c                 between a and b to the maximum magnitude
c                 of all the elements of a
c       errrms -- ratio of the root-mean-square of the elements
c                 of the difference of a and b to the root-mean-square
c                 of all the elements of a
c
        implicit none
        integer m,n,j,k
        real*8 errmax,errrms,diff,amax,arss
        complex*16 a(m,n),b(m,n)
c
c
c       Calculate the maximum magnitude amax of the elements of a
c       and the root-sum-square arss of the elements of a.
c
        amax = 0
        arss = 0
c
        do k = 1,n
          do j = 1,m
c
            if(abs(a(j,k)) .gt. amax) amax = abs(a(j,k))
            arss = arss+a(j,k)*conjg(a(j,k))
c
          enddo ! j
        enddo ! k
c
        arss = sqrt(arss)
c
c
c       Calculate the maximum elementwise absolute difference
c       between a and b, as well as the root-sum-square errrms
c       of the elements of the difference of a and b.
c
        errmax = 0
        errrms = 0
c
        do k = 1,n
          do j = 1,m
c
            diff = abs(a(j,k)-b(j,k))
c
            if(diff .gt. errmax) errmax = diff
            errrms = errrms+diff**2
c
          enddo ! j
        enddo ! k
c
        errrms = sqrt(errrms)
c
c
c       Calculate relative errors.
c
        errmax = errmax/amax
        errrms = errrms/arss
c
c
        return
        end
c
c
c
c
        subroutine matveca(m,x,n,y,a,p2,p3,p4)
c
c       applies the adjoint of a to x, obtaining y.
c
c       input:
c       m -- first dimension of a, and length of x
c       x -- vector to which a^* is to be applied
c       n -- second dimension of a, and length of y
c       a -- matrix whose adjoint is to be applied to x
c            in order to create y
c       p2 -- dummy input
c       p3 -- dummy input
c       p4 -- dummy input
c
c       output:
c       y -- product of a^* and x
c
        implicit none
        integer m,n,j,k
        complex*16 a(m,n),p2,p3,p4,x(m),y(n),sum
c
c
        do k = 1,n
c
          sum = 0
c
          do j = 1,m
            sum = sum+conjg(a(j,k))*x(j)
          enddo ! j
c
          y(k) = sum
c
        enddo ! k
c
c
        return
        end
c
c
c
c
        subroutine matvec(n,x,m,y,a,p2,p3,p4)
c
c       applies a to x, obtaining y.
c
c       input:
c       m -- first dimension of a, and length of x
c       x -- vector to which a is to be applied
c       n -- second dimension of a, and length of y
c       a -- matrix to be applied to x in order to create y
c       p2 -- dummy input
c       p3 -- dummy input
c       p4 -- dummy input
c
c       output:
c       y -- product of a and x
c
        implicit none
        integer m,n,j,k
        complex*16 a(m,n),p2,p3,p4,x(n),y(m),sum
c
c
        do j = 1,m
c
          sum = 0
c
          do k = 1,n
            sum = sum+a(j,k)*x(k)
          enddo ! k
c
          y(j) = sum
c
        enddo ! j
c
c
        return
        end
c
c
c
c
        subroutine zreconsvd(m,krank,u,s,n,v,a)
c
c       forms a = u diag(s) v^*.
c
c       input:
c       m -- first dimension of u and a
c       krank -- size of s, and second dimension of u and v
c       u -- leftmost matrix in the product a = u diag(s) v^*
c       s -- entries on the diagonal in the middle matrix
c            in the product a = u diag(s) v^*
c       n -- second dimension of a and first dimension of v
c       v -- rightmost matrix in the product a = u diag(s) v^*
c
c       output:
c       a -- matrix product u diag(s) v^*
c
        implicit none
        integer m,n,krank,j,k,l
        real*8 s(krank)
        complex*16 u(m,krank),v(n,krank),a(m,n),sum
c
c
        do k = 1,n
          do j = 1,m
c
            sum = 0
c
            do l = 1,krank
              sum = sum+u(j,l)*s(l)*conjg(v(k,l))
            enddo ! l
c
            a(j,k) = sum
c
          enddo ! j
        enddo ! k
c
c
        return
        end
c
c
c
c
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c
c       The above code is for testing and debugging; the remainder of
c       this file contains the following user-callable routines:
c
c
c       routine idzp_rsvd computes the SVD, to a specified precision,
c       of a matrix specified by routines for applying the matrix
c       and its adjoint to arbitrary vectors.
c       This routine is randomized.
c
c
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c
c
c
        subroutine idzp_rsvd(lw,eps,m,n,matveca,p1t,p2t,p3t,p4t,
     1                       matvec,p1,p2,p3,p4,krank,iu,iv,is,w,ier)
c
c       constructs a rank-krank SVD  U Sigma V^*  approximating a
c       to precision eps, where matveca is a routine which applies a^*
c       to an arbitrary vector, and matvec is a routine
c       which applies a to an arbitrary vector; U is an m x krank
c       matrix whose columns are orthonormal, V is an n x krank
c       matrix whose columns are orthonormal, and Sigma is a diagonal
c       krank x krank matrix whose entries are all nonnegative.
c       The entries of U are stored in w, starting at w(iu);
c       the entries of V are stored in w, starting at w(iv).
c       The diagonal entries of Sigma are stored in w,
c       starting at w(is). This routine uses a randomized algorithm.
c
c       input:
c       lw -- maximum usable length (in complex*16 elements)
c             of the array w
c       eps -- precision of the desired approximation
c       m -- number of rows in a
c       n -- number of columns in a 
c       matveca -- routine which applies the adjoint
c                  of the matrix to be SVD'd
c                  to an arbitrary vector; this routine must have
c                  a calling sequence of the form
c
c                  matveca(m,x,n,y,p1t,p2t,p3t,p4t),
c
c                  where m is the length of x,
c                  x is the vector to which the adjoint
c                  of the matrix is to be applied,
c                  n is the length of y,
c                  y is the product of the adjoint of the matrix and x,
c                  and p1t, p2t, p3t, and p4t are user-specified
c                  parameters
c       p1t -- parameter to be passed to routine matveca
c       p2t -- parameter to be passed to routine matveca
c       p3t -- parameter to be passed to routine matveca
c       p4t -- parameter to be passed to routine matveca
c       matvec -- routine which applies the matrix to be SVD'd
c                 to an arbitrary vector; this routine must have
c                 a calling sequence of the form
c
c                 matvec(n,x,m,y,p1,p2,p3,p4),
c
c                 where n is the length of x,
c                 x is the vector to which the matrix is to be applied,
c                 m is the length of y,
c                 y is the product of the matrix and x,
c                 and p1, p2, p3, and p4 are user-specified parameters
c       p1 -- parameter to be passed to routine matvec
c       p2 -- parameter to be passed to routine matvec
c       p3 -- parameter to be passed to routine matvec
c       p4 -- parameter to be passed to routine matvec
c
c       output:
c       krank -- rank of the SVD constructed
c       iu -- index in w of the first entry of the matrix
c             of orthonormal left singular vectors of a
c       iv -- index in w of the first entry of the matrix
c             of orthonormal right singular vectors of a
c       is -- index in w of the first entry of the array
c             of singular values of a; the singular values are stored
c             as complex*16 numbers whose imaginary parts are zeros
c       w -- array containing the singular values and singular vectors
c            of a; w doubles as a work array, and so must be at least
c            (krank+1)*(3*m+5*n+11)+8*krank^2 complex*16 elements long,
c            where krank is the rank returned by the present routine
c       ier -- 0 when the routine terminates successfully;
c              -1000 when lw is too small;
c              other nonzero values when idz_id2svd bombs
c
c       _N.B._: w must be at least (krank+1)*(3*m+5*n+11)+8*krank**2
c               complex*16 elements long, where krank is the rank
c               returned by the present routine. Also, the algorithm
c               used by the present routine is randomized.
c
        implicit none
        integer m,n,krank,lw,lw2,ilist,llist,iproj,icol,lcol,lp,
     1          iwork,lwork,ier,lproj,iu,iv,is,lu,lv,ls,iui,ivi,isi,k
        real*8 eps
        complex*16 p1t,p2t,p3t,p4t,p1,p2,p3,p4,w(*)
        external matveca,matvec
c
c
c       Allocate some memory.
c
        lw2 = 0
c
        ilist = lw2+1
        llist = n
        lw2 = lw2+llist
c
        iproj = lw2+1
c
c
c       ID a.
c
        lp = lw-lw2
        call idzp_rid(lp,eps,m,n,matveca,p1t,p2t,p3t,p4t,krank,
     1                w(ilist),w(iproj),ier)
        if(ier .ne. 0) return
c
c
        if(krank .gt. 0) then
c
c
c         Allocate more memory.
c
          lproj = krank*(n-krank)
          lw2 = lw2+lproj
c
          icol = lw2+1
          lcol = m*krank
          lw2 = lw2+lcol
c
          iui = lw2+1
          lu = m*krank
          lw2 = lw2+lu
c
          ivi = lw2+1
          lv = n*krank
          lw2 = lw2+lv
c
          isi = lw2+1
          ls = krank
          lw2 = lw2+ls
c
          iwork = lw2+1
          lwork = (krank+1)*(m+3*n+10)+9*krank**2
          lw2 = lw2+lwork
c
c
          if(lw .lt. lw2) then
            ier = -1000
            return
          endif
c
c
          call idzp_rsvd0(m,n,matveca,p1t,p2t,p3t,p4t,
     1                    matvec,p1,p2,p3,p4,krank,w(iui),w(ivi),
     2                    w(isi),ier,w(ilist),w(iproj),w(icol),
     3                    w(iwork))
          if(ier .ne. 0) return
c
c
          iu = 1
          iv = iu+lu
          is = iv+lv
c
c
c         Copy the singular values and singular vectors
c         into their proper locations.
c
          do k = 1,lu
            w(iu+k-1) = w(iui+k-1)
          enddo ! k 
c
          do k = 1,lv
            w(iv+k-1) = w(ivi+k-1)
          enddo ! k
c
          call idz_reco(ls,w(isi),w(is))
c
c
        endif ! krank .gt. 0
c
c
        return
        end
c
c
c
c
        subroutine idzp_rsvd0(m,n,matveca,p1t,p2t,p3t,p4t,
     1                        matvec,p1,p2,p3,p4,krank,u,v,s,ier,
     2                        list,proj,col,work)
c
c       routine idzp_rsvd serves as a memory wrapper
c       for the present routine (please see routine idzp_rsvd
c       for further documentation).
c
        implicit none
        integer m,n,krank,list(n),ier
        real*8 s(krank)
        complex*16 p1t,p2t,p3t,p4t,p1,p2,p3,p4,u(m,krank),v(n,krank),
     1             proj(krank,n-krank),col(m*krank),
     2             work((krank+1)*(m+3*n+10)+9*krank**2)
        external matveca,matvec
c
c
c       Collect together the columns of a indexed by list into col.
c
        call idz_getcols(m,n,matvec,p1,p2,p3,p4,krank,list,col,work)
c
c
c       Convert the ID to an SVD.
c
        call idz_id2svd(m,krank,col,n,list,proj,u,v,s,ier,work)
c
c
        return
        end
c
c
c
c
        subroutine idz_reco(n,a,b)
c
c       copies the real*8 array a into the complex*16 array b.
c
c       input:
c       n -- length of a and b
c       a -- real*8 array to be copied into b
c
c       output:
c       b -- complex*16 copy of a
c
        integer n,k
        real*8 a(n)
        complex*16 b(n)
c
c
        do k = 1,n
          b(k) = a(k)
        enddo ! k
c
c
        return
        end
