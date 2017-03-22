for p in ("Knet","ArgParse","Compat","GZip")
    Pkg.installed(p) == nothing && Pkg.add(p)
end
using Knet,ArgParse,Compat,GZip

function generateData(upperbound,token)
	x=rand(1:9999999, upperbound,1);
	y=rand(1:9999999, upperbound,1);
	if(isequal(token, "+"))
		z=x+y;
	elseif(isequal(token, "*"))
		z=x.*y;
	end
	s= Any[];
	for i= 1: upperbound 
		a=bin(x[i])*token*bin(y[i]); 
		push!(s,a)                       
	end
	return s,z
end
function minibatch(x, y, batchsize)
	data=Any[];
	for i=1:batchsize:size(x,1)-batchsize+1
		push!(data, x[i:i+batchsize-1], y[i:i+batchsize-1])
	end
	return data
	
end
function embeddedMatrix()
#We have four different characters 0, 1, + and *
	vocab = Dict{Char,Int}()
	vocab['0']=1
	vocab['1']=2
	vocab['+']=3
	vocab['*']=4
	return E=rand(4,24), vocab# m=24  
end
#input = string
function init_state(E, input, vocab, w,m=24)
	c=1;k=1
	n=length(input);
	s=zeros(w,n,m)
	while c <= endof(input)
		ch=input[c];
		index= vocab[ch];
		v=E[index]
		s[1,k,:]=v;
		c=nextind(input,c)
		k=k+1
	end
	return s
	
end
function initWeight(kw,kh,x,y, atype=Array{Float32}, winit=0.1, m=24, )
	w = Array(Any,4);
	w[1] = KnetArray{Float32}(winit*randn(kw,kh,m,m));
	w[2] = KnetArray{Float32}(zeros(1,1,m,1));
	X=1+floor(x-kw)
	Y=1+floor(y-kh)
	w[3] = KnetArray{Float32}(randn(X,Y)*winit);
	w[4] = KnetArray{Float32}(zeros(1,1,X,1));
	return w;
end
function CGRU(s,w)
	return s
end

function loss(sn, ygold, m=24)
	y=zeros(length(ygold), m)
	ypred=zeros(size(sn,2),m)# it should not be zero by the way
	c=1;k=1
	while c <= endof(input)
		ch=input[c];
		index= vocab[ch];
		v=E[index]
		y[k,:]=v;
		c=nextind(input,c)
		k=k+1
	end
	
	for k=1:size(sn,2)
	ypred(k,:)=sn[1,k,:]
	end
	ynorm = logp(ypred) # ypred .- log(sum(exp(ypred),1))
    return -sum(y .* ynorm) / size(y,2)
	
end

function train(sn, ygold, gclip,w)
	gloss=lossgradient(sn, ygold[1])
	gloss2= gloss.*gloss
	gnorm=sqrt(sum(gloss2, 1))
	if gnorm >gclip
		gloss=(gloss*gclip)/gnorm
	end
	opts = map(x->Knet.Adam(), w);
	update!(sn,gloss, opts)

end
w=3;
m=24;
kh=3;
kw=3;
gclip=1
x, ygold=generateData(10000, "*");
data=minibatch(x, ygold, 1000)
E, vocab=embeddedMatrix();
s0=init_state(E, x[1], vocab,3 )
n=length(x[1]);
w=initWeight(kh,kw, w, n)
#sn=CGRU(s,w)#upto here it is correst
#lossgradient = grad(loss);
#train(sn, ygold, gclip,w)




